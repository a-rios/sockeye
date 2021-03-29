# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for scoring.
"""
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import inference
from . import model
from . import utils
from . import vocab
from . import loss
from .inference import TranslatorInput, TranslatorOutput
from .output_handler import OutputHandler

logger = logging.getLogger(__name__)


class ScoringModel(model.SockeyeModel):
    """
    ScoringModel is a TrainingModel (which is in turn a SockeyeModel) that scores a pair of sentences.
    That is, it full unrolls over source and target sequences, running the encoder and decoder,
    but stopping short of computing a loss and backpropagating.
    It is analogous to TrainingModel, but more limited.

    :param config: Configuration object holding details about the model.
    :param model_dir: Directory containing the trained model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param score_type: The type of score to output (negative logprob or logprob).
    :param length_penalty: The length penalty instance to use.
    :param brevity_penalty: The brevity penalty instance to use.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 model_dir: str,
                 context: List[mx.context.Context],
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 default_bucket_key: Tuple[int, int],
                 score_type: str,
                 length_penalty: inference.LengthPenalty,
                 brevity_penalty: inference.BrevityPenalty,
                 softmax_temperature: Optional[float] = None,
                 brevity_penalty_type: str = '',
                 constant_length_ratio: float = 0.0,
                 attention_monotonicity_scoring: Optional[bool] = False,
                 attention_monotonicity_scoring_margin: Optional[float] = 0.0,
                 monotonicity_on_heads: Optional[Tuple[int,int]] = None,
                 monotonicity_on_layers: Optional[Tuple[int,int]] = None,
                 checkpoint: Optional[int] = None) -> None:
        super().__init__(config)
        self.context = context
        self.score_type = score_type
        self.length_penalty = length_penalty
        self.brevity_penalty = brevity_penalty
        self.softmax_temperature = softmax_temperature
        self.attention_monotonicity_scoring = attention_monotonicity_scoring
        self.attention_monotonicity_scoring_margin = attention_monotonicity_scoring_margin
        self.monotonicity_on_heads = monotonicity_on_heads
        self.monotonicity_on_layers = monotonicity_on_layers
        self.checkpoint = checkpoint

        if brevity_penalty_type == C.BREVITY_PENALTY_CONSTANT:
            if constant_length_ratio <= 0.0:
                self.constant_length_ratio = self.length_ratio_mean
                logger.info("Using constant length ratio saved in the model config: %f",
                            self.constant_length_ratio)
        else:
            self.constant_length_ratio = -1.0

        # Create the computation graph
        self._initialize(provide_data, provide_label, default_bucket_key)

        # Load model parameters into graph
        params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        if self.checkpoint is not None:
            params_fname = os.path.join(model_dir, C.PARAMS_NAME % self.checkpoint)
        super().load_params_from_file(params_fname)
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=False)

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]) -> None:
        """
        Initializes model components, creates scoring symbol and module, and binds it.

        :param provide_data: List of data descriptors.
        :param provide_label: List of label descriptors.
        :param default_bucket_key: The default maximum (source, target) lengths.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        target_words = mx.sym.reshape(data=target, shape=(-1,), name="target_words")

        # labels shape: (batch_size, target_length) (usually the maximum target sequence length)
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)

        data_names = [C.SOURCE_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        self.attention_monotonicity_scorer=None
        if self.attention_monotonicity_scoring:
            self.attention_monotonicity_scorer = loss.MonotoneAttention(self.config.attention_monotonicity_config_loss)

        def sym_gen(seq_lens):
            """
            Returns a (grouped) symbol containing the summed score for each sentence, as well as the entire target
            distributions for each word.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
            source_encoded_length,
            source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                          source_embed_length,
                                                          source_embed_seq_len,
                                                          source=source_words,
                                                          separator_id=self.config.separator_id)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded, pointer_scores, attention_scores_list = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(mx.sym.reshape(data=target_decoded, shape=(-3, 0)))
            if self.config.num_pointers:
                logits = mx.sym.concat(logits, pointer_scores, dim=1)
            
            # logits after reshape: (batch_size, target_seq_len, target_vocab_size)
            logits = mx.sym.reshape(data=logits, shape=(-4, -1, target_embed_seq_len, 0))

            if self.softmax_temperature is not None:
                logits = logits / self.softmax_temperature

            # Compute the softmax along the final dimension.
            # target_dists: (batch_size, target_seq_len, target_vocab_size)
            target_dists = mx.sym.softmax(data=logits, axis=2, name=C.SOFTMAX_NAME)

            # Select the label probability, then take their logs.
            # probs and scores: (batch_size, target_seq_len)
            probs = mx.sym.pick(target_dists, labels)
            scores = mx.sym.log(probs)
            if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
                scores = -1 * scores

            # Sum, then apply length penalty. The call to `mx.sym.where` masks out invalid values from scores.
            # zeros and sums: (batch_size,)
            zeros = mx.sym.zeros_like(scores)
            sums = mx.sym.sum(mx.sym.where(labels != 0, scores, zeros), axis=1) / (self.length_penalty(target_length - 1))

            # Deal with the potential presence of brevity penalty
            # length_ratio: (batch_size,)
            if self.constant_length_ratio > 0.0:
                # override all ratios with the constant value
                length_ratio = self.constant_length_ratio * mx.sym.ones_like(sums)
            else:
                # predict length ratio if supported
                length_ratio = self.length_ratio(source_encoded, source_encoded_length).reshape((-1,)) \
                                    if self.length_ratio is not None else mx.sym.zeros_like(sums)
            sums = sums - self.brevity_penalty(target_length - 1, length_ratio * source_encoded_length)

            if self.attention_monotonicity_scorer is not None:
                num_attention_heads = 1
                if hasattr(self.config.config_decoder, "attention_heads"):
                    num_attention_heads = self.config.config_decoder.attention_heads
                attention_monotonicity_scores, avg_position_increase = self.attention_monotonicity_scorer.get_loss(attention_scores_list=attention_scores_list,
                                                                                    num_attention_heads=num_attention_heads,
                                                                                    target_words=target_words,
                                                                                    source_words=source_words,
                                                                                    source_length=source_length,
                                                                                    target_length=target_length,
                                                                                    margin=self.attention_monotonicity_scoring_margin,
                                                                                    monotonicity_on_heads=self.monotonicity_on_heads,
                                                                                    monotonicity_on_layers=self.monotonicity_on_layers,
                                                                                    grad_scale=1.0,
                                                                                    return_position_diffs=True)

                return mx.sym.Group([sums, target_dists, attention_monotonicity_scores, avg_position_increase] + attention_scores_list ), data_names, label_names
            # Return the sums and the target distributions
            # sums: (batch_size,) target_dists: (batch_size, target_seq_len, target_vocab_size)
            return mx.sym.Group([sums, target_dists]), data_names, label_names

        symbol, _, __ = sym_gen(default_bucket_key)
        self.module = mx.mod.Module(symbol=symbol,
                                    data_names=data_names,
                                    label_names=label_names,
                                    logger=logger,
                                    context=self.context)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=False,
                         force_rebind=False,
                         grad_req='null')

    def run(self, batch: mx.io.DataBatch) -> List[mx.nd.NDArray]:
        """
        Runs the forward pass and returns the outputs.

        :param batch: The batch to run.
        :return: The grouped symbol (probs and target dists) and lists containing the data names and label names.
        """
        self.module.forward(batch, is_train=False)
        return self.module.get_outputs()


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel sentences.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param source_vocabs: The source vocabularies.
    :param target_vocab: The target vocabulary.
    """
    def __init__(self,
                 model: ScoringModel,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 constant_length_ratio: float = -1.0) -> None:
        self.source_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv = vocab.reverse_vocab(target_vocab)
        self.model = model
        self.exclude_list = {source_vocabs[0][C.BOS_SYMBOL], target_vocab[C.EOS_SYMBOL], C.PAD_ID}
        self.constant_length_ratio = constant_length_ratio

    def score(self,
              score_iter,
              output_handler: OutputHandler,
              attention_handler: Optional[OutputHandler] = None):

        total_time = 0.
        sentence_no = 0
        batch_no = 0
        for batch_no, batch in enumerate(score_iter, 1):
            batch_tic = time.time()

            # Run the model and get the outputs
            output = self.model.run(batch)
            scores = output[0]
            attention_monotonicity_scores = mx.nd.zeros_like(scores)
            if self.model.attention_monotonicity_scoring:
                attention_monotonicity_scores = output[2]
                avg_position_increase = output[3]
                num_attention_heads = 1
                if hasattr(self.model.config.config_decoder, "attention_heads"):
                    num_attention_heads = self.model.config.config_decoder.attention_heads
                start_layer, end_layer = (1,1)
                if self.model.monotonicity_on_layers is not None:
                    start_layer, end_layer = self.model.monotonicity_on_layers
                layer_attention_list = []
                for i in range(start_layer+3, end_layer+4): ## shift, output[2]=mono score, output[3] = position_diffs
                    layer_attention_list.append(output[i])

            batch_time = time.time() - batch_tic
            total_time += batch_time

            batch_size = len(batch.data[0])

            for sentno, (source, target, score, monotonicity_score, attention_percentage_position_increase) in enumerate(zip(batch.data[0], batch.data[1], scores, attention_monotonicity_scores, avg_position_increase), 1):

                # The last batch may be underfilled, in which case batch.pad will be set
                if sentno > (batch_size - batch.pad):
                    break

                sentence_no += 1

                # Transform arguments in preparation for printing
                source_ids = [int(x) for x in source[:, 0].asnumpy().tolist()]
                source_tokens = list(data_io.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list))
                target_ids = [int(x) for x in target.asnumpy().tolist()]
                target_string = C.TOKEN_SEPARATOR.join(
                    data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))

                # Report a score of -inf for invalid sentence pairs (empty source and/or target)
                if source[0][0] == C.PAD_ID or target[0] == C.PAD_ID:
                    score = -np.inf
                    monotonicity_score = -np.inf
                    attention_percentage_position_increase = -np.inf
                else:
                    score = score.asscalar()
                    monotonicity_score = monotonicity_score.asscalar()
                    attention_percentage_position_increase = attention_percentage_position_increase.asscalar()

                # Output handling routines require us to make use of inference classes.
                output_handler.handle(TranslatorInput(sentence_no, source_tokens),
                                      TranslatorOutput(sentence_no, target_string, None, None, score, None, None, None, None, None, None, monotonicity_score, attention_percentage_position_increase),
                                      batch_time)

                if attention_handler is not None:
                    target_tokens = list(data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))
                    target_tokens.insert(0, "BOS")
                    source_tokens.append("EOS")
                    #print(layer_attention_list)
                    for layer_attention, i in zip(layer_attention_list, range(start_layer, end_layer+1)):
                        if num_attention_heads > 1:
                            # shape (batch * head, trg_len, src_len)
                            # attention matrix for each head for this sample
                            start = 1
                            end = num_attention_heads
                            if self.model.monotonicity_on_heads is not None:
                                start, end = self.model.monotonicity_on_heads
                            layer_attention = layer_attention.reshape(shape=(-4, -1, num_attention_heads, -2))
                            for head in range(start-1, end):
                                attention = layer_attention[(sentno-1)][head] ## trg_len, src_len, sentno starts at 1, need index to start at 0
                                attention = attention.slice_axis(axis=0, begin=0, end=(len(target_tokens)) )
                                attention = attention.slice_axis(axis=1, begin=0, end=(len(source_tokens)) )
                                attention_handler.handle(t_input=TranslatorInput(sentence_no, source_tokens),
                                              t_output=TranslatorOutput(sentence_no, target_string, target_tokens, attention.asnumpy(), score, None, None, None, None, None, None, monotonicity_score),
                                              layer=i,
                                              head=head+1)
                        else:
                            attention = layer_attention[sentno-1] ## trg_len, src_len
                            attention = attention.slice_axis(axis=0, begin=0, end=(len(target_tokens)) )
                            attention = attention.slice_axis(axis=1, begin=0, end=(len(source_tokens)) )
                            attention_handler.handle(t_input=TranslatorInput(sentence_no, source_tokens),
                                              t_output=TranslatorOutput(sentence_no, target_string, target_tokens, attention.asnumpy(), score, None, None, None, None, None, None, monotonicity_score),
                                              layer=i)

        if sentence_no != 0:
            logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                        sentence_no, math.ceil(sentence_no / batch_no), total_time,
                        total_time / sentence_no, sentence_no / total_time)
        else:
            logger.info("Processed 0 lines.")
