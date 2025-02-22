# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from typing import List

import torch
from megatron.core import tensor_parallel
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.models.gpt import GPTModel

from cosmos1.models.autoregressive.modules.embedding import SinCosPosEmbAxisTE


class CosmosInferenceWrapper(GPTInferenceWrapper):
    def __init__(self, model: GPTModel, args: Namespace, config):
        super().__init__(model, args)
        self.config = config

    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        super().prep_model_for_inference(prompts_tokens=prompts_tokens)
        self.abs_pos_emb = self._initialize_abs_pos_emb()

    def _initialize_abs_pos_emb(self):
        pos_emb = SinCosPosEmbAxisTE(dim=self.config.hidden_size, latent_shape=[5, 40, 64], pad_to_multiple_of=64)
        training_type = "text_to_video"
        abs_pos_emb = pos_emb.forward(training_type=training_type)
        abs_pos_emb = abs_pos_emb.transpose(0, 1).contiguous()
        return abs_pos_emb

    def get_batch_for_context_window(self, context_start_position: int, context_end_position: int) -> List:
        data_at_step_idx = super().get_batch_for_context_window(context_start_position, context_end_position)
        absposembed2use = self.abs_pos_emb[context_start_position:context_end_position, :, :]
        data_at_step_idx.append(absposembed2use)
        return data_at_step_idx

    def set_context_tokens(self, batch_context_tokens):
        self.context_tokens = batch_context_tokens

    def forward_pass_without_pipeline_parallel(self, inference_input: List) -> torch.Tensor:
        tokens, position_ids, attention_mask, abs_pos_embed = inference_input
        assert hasattr(
            self, "context_tokens"
        ), "Expected to have context tokens. Not present. Call set_context_tokens with the encoder embeddings"
        extra_block_kwargs = {"context": self.context_tokens, "extra_positional_embeddings": abs_pos_embed}
        packed_seq_params = None

        logits = self.model(
            tokens,
            position_ids,
            attention_mask,
            inference_params=self.inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        self.inference_params.sequence_len_offset += tokens.size(1)

        return logits


class CosmosTextGenerationController(SimpleTextGenerationController):
    def __init(self, inference_wrapped_model: CosmosInferenceWrapper, tokenizer):
        super().__init__(inference_wrapped_model, tokenizer)

    def generate_all_output_tokens_static_batch(self, active_requests):
        batch_context_tokens = (
            list(map(lambda request: request.encoder_prompt, active_requests.values()))[0]
            .to(torch.bfloat16)
            .permute(1, 0, 2)
        )
        self.inference_wrapped_model.set_context_tokens(batch_context_tokens)
        active_requests = super().generate_all_output_tokens_static_batch(active_requests)
        return active_requests
