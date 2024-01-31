# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""
"--version 3.4.0"

import pdb
import os
from sys import prefix
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.models.gpt2 import GPT2Config

# from modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        try:
            if config.is_contrastive_prefix:
                assert config.prefix_len is not None
                assert config.prefix_mid_size is not None
                assert config.map_dict is not None
                assert config.label_keys is not None
                self.prefix_len = config.prefix_len
                self.prefix_mid_size = config.prefix_mid_size
                self.map_dict = config.map_dict
                self.label_keys = config.label_keys

                self.prefix_keys_embeddings = nn.ModuleDict()
                self.prefix_values_embeddings = nn.ModuleDict()
                self.prefix_mlp = nn.ModuleDict()
                for key in config.label_keys:
                    self.prefix_keys_embeddings[key] = nn.ModuleDict()
                    self.prefix_values_embeddings[key] = nn.ModuleDict()
                    self.prefix_mlp[key] = nn.ModuleDict()
                    for i in range(config.map_dict[key]['dim']):
                        # pdb.set_trace()
                        i = str(i)
                        self.prefix_keys_embeddings[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.prefix_mid_size)
                        self.prefix_values_embeddings[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.prefix_mid_size)
                        self.prefix_mlp[key][config.map_dict[key][i]] = Conv1D(config.n_embd, config.prefix_mid_size)
            else:
                pass
        except:
            pass

        try:
            if config.is_promptgating:
                assert config.prefix_len is not None
                assert config.map_dict is not None
                assert config.prefix_mid_size is not None
                assert config.label_keys is not None
                self.prefix_len = config.prefix_len
                self.prefix_mid_size = config.prefix_mid_size
                self.map_dict = config.map_dict
                self.label_keys = config.label_keys

                self.prefix_keys_embeddings = nn.ModuleDict()
                self.prefix_values_embeddings = nn.ModuleDict()
                self.prefix_keys_gate = nn.ModuleDict()
                self.prefix_values_gate = nn.ModuleDict()
                self.prefix_mlp = nn.ModuleDict()
                for key in config.label_keys:
                    self.prefix_keys_embeddings[key] = nn.ModuleDict()
                    self.prefix_values_embeddings[key] = nn.ModuleDict()
                    self.prefix_keys_gate[key] = nn.ModuleDict()
                    self.prefix_values_gate[key] = nn.ModuleDict()
                    self.prefix_mlp[key] = nn.ModuleDict()
                    for i in range(config.map_dict[key]['dim']):
                        self.prefix_keys_embeddings[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.prefix_mid_size)
                        self.prefix_values_embeddings[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.prefix_mid_size)
                        self.prefix_mlp[key][config.map_dict[key][i]] = Conv1D(config.n_embd, config.prefix_mid_size)
                        self.prefix_keys_gate[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.n_embd)
                        self.prefix_values_gate[key][config.map_dict[key][i]] = nn.Embedding(config.prefix_len, config.n_embd)
            else:
                pass
        except:
            pass

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        use_prefix=False,
        prefix_ids_pg=None, # [{key: value}, ] eg. [{'sentiment': 'Positive'}] used for prompt_gating
        prefix_ids_cp=None, # [{key: value}, ] eg. [{'sentiment': 'Positive'}] used for contrastive_prefix
    ):
        device = hidden_states.device
        batch_size = hidden_states.shape[0]

        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)
        
        if prefix_ids_cp is not None:
            prefix_index = torch.tensor(range(self.prefix_len)).to(device)
            prefix_index = prefix_index.unsqueeze(0)
            batch_prefix_keys_embeddings = None
            batch_prefix_values_embeddings = None
            for ids in prefix_ids_cp:
                prefix_keys_embeddings = None
                prefix_values_embeddings = None
                keys = list(ids.keys())
                for _key in keys:
                    _value = ids[_key]
                    if prefix_keys_embeddings is None:
                        prefix_keys_embeddings = self.prefix_keys_embeddings[_key][_value](prefix_index)
                        prefix_keys_embeddings = self.prefix_mlp[_key][_value](prefix_keys_embeddings)

                        prefix_values_embeddings = self.prefix_values_embeddings[_key][_value](prefix_index)
                        prefix_values_embeddings = self.prefix_mlp[_key][_value](prefix_values_embeddings)
                    else:
                        temp_keys = self.prefix_keys_embeddings[_key][_value](prefix_index)
                        temp_keys = self.prefix_mlp[_key][_value](temp_keys)

                        temp_values = self.prefix_values_embeddings[_key][_value](prefix_index)
                        temp_values = self.prefix_mlp[_key][_value](temp_values)

                        prefix_keys_embeddings = torch.cat([prefix_keys_embeddings, temp_keys], dim=1)
                        prefix_values_embeddings = torch.cat([prefix_values_embeddings, temp_values], dim=1)

                if batch_prefix_keys_embeddings is None:
                    batch_prefix_keys_embeddings = prefix_keys_embeddings
                    batch_prefix_values_embeddings = prefix_values_embeddings
                else:
                    batch_prefix_keys_embeddings = torch.cat([batch_prefix_keys_embeddings, prefix_keys_embeddings], dim=0)
                    batch_prefix_values_embeddings = torch.cat([batch_prefix_values_embeddings, prefix_values_embeddings], dim=0)
                
            batch_prefix_keys_embeddings = self.split_heads(batch_prefix_keys_embeddings, k=True)
            batch_prefix_values_embeddings = self.split_heads(batch_prefix_values_embeddings)

            key = torch.cat((batch_prefix_keys_embeddings, key), dim=-1)
            value = torch.cat((batch_prefix_values_embeddings, value), dim=-2)

        
        if prefix_ids_pg is not None:
            prefix_index = torch.tensor(range(self.prefix_len)).to(device)
            prefix_index = prefix_index.unsqueeze(0)
            batch_prefix_keys_embeddings = None
            batch_prefix_values_embeddings = None
            for ids in prefix_ids_pg:
                prefix_keys_embeddings = None
                prefix_values_embeddings = None
                keys = list(ids.keys())
                for _key in keys:
                    _value = ids[_key]
                    if prefix_keys_embeddings is None:
                        prefix_keys_embeddings = self.prefix_keys_embeddings[_key][_value](prefix_index)
                        prefix_keys_embeddings = self.prefix_mlp[_key][_value](prefix_keys_embeddings)
                        prefix_keys_gate = self.prefix_keys_gate[_key][_value](prefix_index)
                        prefix_keys_embeddings = prefix_keys_embeddings * torch.sigmoid(prefix_keys_gate)

                        prefix_values_embeddings = self.prefix_values_embeddings[_key][_value](prefix_index)
                        prefix_values_embeddings = self.prefix_mlp[_key][_value](prefix_values_embeddings)
                        prefix_values_gate = self.prefix_values_gate[_key][_value](prefix_index)
                        prefix_values_embeddings = prefix_values_embeddings * torch.sigmoid(prefix_values_gate)
                    else:
                        temp_keys = self.prefix_keys_embeddings[_key][_value](prefix_index)
                        temp_keys = self.prefix_mlp[_key][_value](temp_keys)
                        temp_keys_gate = self.prefix_keys_gate[_key][_value](prefix_index)
                        temp_keys = temp_keys * torch.sigmoid(temp_keys_gate)

                        temp_values = self.prefix_values_embeddings[_key][_value](prefix_index)
                        temp_values = self.prefix_mlp[_key][_value](temp_values)
                        temp_values_gate = self.prefix_values_gate[_key][_value](prefix_index)
                        temp_values = temp_values * torch.sigmoid(temp_values_gate)

                        prefix_keys_embeddings = torch.cat([prefix_keys_embeddings, temp_keys], dim=1)
                        prefix_values_embeddings = torch.cat([prefix_values_embeddings, temp_values], dim=1)
                
                if batch_prefix_keys_embeddings is None:
                    batch_prefix_keys_embeddings = prefix_keys_embeddings
                    batch_prefix_values_embeddings = prefix_values_embeddings
                else:
                    batch_prefix_keys_embeddings = torch.cat([batch_prefix_keys_embeddings, prefix_keys_embeddings], dim=0)
                    batch_prefix_values_embeddings = torch.cat([batch_prefix_values_embeddings, prefix_values_embeddings], dim=0)
            # pdb.set_trace()
            
            batch_prefix_keys_embeddings = self.split_heads(batch_prefix_keys_embeddings, k=True)
            batch_prefix_values_embeddings = self.split_heads(batch_prefix_values_embeddings)

            key = torch.cat((batch_prefix_keys_embeddings, key), dim=-1)
            value = torch.cat((batch_prefix_values_embeddings, value), dim=-2)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        use_prefix=False,
        prefix_ids_pg=None,
        prefix_ids_cp=None,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            use_prefix=use_prefix,
            prefix_ids_pg=prefix_ids_pg,
            prefix_ids_cp=prefix_ids_cp,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (cross_attentions, attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

        try:
            if config.prompt_config and config.prompt_len:
                self.prompt_len = config.prompt_len
                self.prompt_config = config.prompt_config
                self.prompt_embeddings = nn.ModuleDict()
                for key in list(config.prompt_config.keys()):
                    self.prompt_embeddings[key] = nn.ModuleDict()
                    values = config.prompt_config[key]
                    for value in values:
                        self.prompt_embeddings[key][value] = nn.Embedding(config.prompt_len, config.n_embd)
            else:
                pass
        except:
            pass

        try:
            if config.is_tailer and config.MAP_len:
                self.prompt_MAP_connector = nn.Embedding(config.MAP_len, config.n_embd)
                self.MAP_len = config.MAP_len
            else:
                pass
        except:
            pass
        
        try:
            if config.is_dcg:
                assert config.dcg_att_num is not None
                assert config.dcg_att_len is not None
                assert config.dcg_task_len is not None
                self.dcg_mlp = nn.Sequential(
                    nn.Linear(config.dcg_att_num * config.n_embd, config.dcg_att_len * config.n_embd)
                )
                self.dcg_prompt_embeddings = nn.Embedding(config.dcg_task_len, config.n_embd)
            else:
                pass
        except:
            pass

        try:
            if config.is_fudge:
                assert config.fudge_class_size is not None
                self.fudge_mlp = torch.nn.Linear(config.n_embd, config.fudge_class_size)
            else:
                pass
        except:
            pass

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="gpt2",
    #     output_type=BaseModelOutputWithPast,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_prefix=False,
        use_prompt=False,
        prompt_ids=None,    # [[..], [..], ..] len(prompt_ids)=batch_size,
        prefix_ids_pg=None,
        prefix_ids_cp=None,
        config=None,
        att_tokens_ids=None,    # torch.tensor([torch.tensor(idx1,idx2,..), ..]) batch_size * dcg_att_num
        use_MAP=None,
        **kwargs,
    ):
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if use_prompt, setting prompt_len
        if use_prompt and past_key_values is None:
            assert self.prompt_config is not None
            assert self.prompt_len is not None
            assert prompt_ids is not None
            prompt_index = torch.tensor(range(self.prompt_len)).to(device)
            prompt_index = prompt_index.unsqueeze(0)
            prompt_embeddings_dict = dict()
            prompt_embeds = None
            for i in range(batch_size):
                prompt_embeddings_dict['idx_{}'.format(i)] = dict()
                for key in list(prompt_ids.keys()):
                    value = prompt_ids[key][i]
                    prompt_embeddings_dict['idx_{}'.format(i)][key] = self.prompt_embeddings[key][value](prompt_index)
            
            for i in range(batch_size):
                temp = None
                for key in list(prompt_ids.keys()):
                    if temp == None:
                        temp = prompt_embeddings_dict['idx_{}'.format(i)][key]
                    else:
                        temp = torch.cat([temp, prompt_embeddings_dict['idx_{}'.format(i)][key]], dim=1)
                if prompt_embeds == None:
                    prompt_embeds = temp
                else:
                    prompt_embeds = torch.cat([prompt_embeds, temp], dim=0)
                del temp
                
            prompt_len = len(prompt_ids) * self.prompt_len

            if use_MAP:
                assert self.config.is_tailer is not None
                MAP_index = torch.tensor(range(self.MAP_len)).unsqueeze(0).to(device)
                MAP_embeds = self.prompt_MAP_connector(MAP_index).expand(batch_size, self.MAP_len, -1)

                prompt_embeds = torch.cat([prompt_embeds, MAP_embeds], dim=1)
                prompt_len = len(prompt_ids) * self.prompt_len + self.MAP_len
            
        else:
            prompt_len = 0
        
        # dcg only:
        try:
            if config.is_dcg and past_key_values is None:
                assert config.dcg_att_num is not None
                assert config.dcg_att_len is not None
                assert config.dcg_task_len is not None
                assert att_tokens_ids is not None
                att_tokens_embeds = self.wte(att_tokens_ids)    # batch_size * dcg_att_num * n_embd
                att_tokens_embeds = att_tokens_embeds.view(batch_size, -1)  # batch_size * (dcg_att_num * n_embd)
                att_prompt = self.dcg_mlp(att_tokens_embeds)    # batch_size * (dcg_att_len * n_embd)
                att_prompt = att_prompt.view(batch_size, config.dcg_att_len, config.n_embd) # batch_size * dcg_att_len, n_embd
                task_index = torch.tensor(range(config.dcg_task_len)).to(device)
                task_index = task_index.unsqueeze(0).expand(batch_size, config.dcg_task_len)
                task_prompt = self.dcg_prompt_embeddings(task_index)    # batch_size * dcg_task_len * n_embd

                prompt_embeds = torch.cat([att_prompt, task_prompt], dim=1)
                prompt_len = config.dcg_att_len + config.dcg_task_len
            else:
                prompt_len = 0
        except:
            pass

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        # pdb.set_trace()
        if position_ids is None:
            # adding prompt_len position if use_prompt
            if past_length == 0:
                position_ids = torch.arange(0, prompt_len + input_shape[-1], dtype=torch.long, device=device)
            else:
                position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, prompt_len + input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # if use_prompt, add prompt_mask to attention_mask
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        # if use_prompt, cat inputs_embeds and prompt_embeds
        # as there is an eos_token_embeds at the beginning of inputs_embeds, we need
        # to insert the prompt_embeds between the eos_token_embeds and the other
        # part of inputs_embeds
        if past_length == 0:
            try:
                if use_prompt or config.is_dcg:
                    # eos token set before prompt and inputs
                    temp = torch.cat([inputs_embeds[:, :1, :], prompt_embeds], dim=1)
                    inputs_embeds = torch.cat([temp, inputs_embeds[:, 1:, :]], dim=1)
                    # eos token set between prompt and inputs
                    # temp = torch.cat([prompt_embeds, inputs_embeds[:, :1, :]], dim=1)
                    # inputs_embeds = torch.cat([temp, inputs_embeds[:, 1:, :]], dim=1)
            except:
                pass
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        
        try:
            if use_prompt or config.is_dcg:
                output_shape = hidden_states.shape
            else:
                output_shape = input_shape + (hidden_states.size(-1),)
        except:
            output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    use_prefix=use_prefix,
                    prefix_ids_pg=prefix_ids_pg,
                    prefix_ids_cp=prefix_ids_cp,
                )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)
        
        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            # att_embeds=att_prompt,
        )


class GPT2LMHeadModel(GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_prefix=False,
        use_prompt=False,
        prompt_ids=None,
        prefix_ids_pg=None,
        prefix_ids_cp=None,
        config=None,
        att_tokens_ids=None,
        use_MAP=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_prefix=use_prefix,
            use_prompt=use_prompt,
            prompt_ids=prompt_ids,
            prefix_ids_pg=prefix_ids_pg,
            prefix_ids_cp=prefix_ids_cp,
            config=config,
            att_tokens_ids=att_tokens_ids,
            use_MAP=use_MAP,
        )
        hidden_states = transformer_outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.last_hidden_state,
            attentions=transformer_outputs.attentions,
            # att_embeds=transformer_outputs.att_embeds,
        )
