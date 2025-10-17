from vllm.model_executor.models.gpt_oss import OAIAttention, TransformerBlock, GptOssModel, GptOssForCausalLM
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.model_executor.models.qwen3_moe import IntermediateTensors

from utils.utils import topk_indices

import os
import numpy as np
import torch

from typing import Optional, Union 

def steering_attention_forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int=None,
    steering_flag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    t = self.norm(hidden_states)

    qkv, _ = self.qkv(t)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    v = v.contiguous()
    attn_output = self.attn(q, k, v)
    
    if self.steering_mode == "before_o_proj":
        if  steering_flag.sum() > 0 and \
            layer_idx in self.valid_hyperplane_vector_dict.keys():
            tp_rank = get_tensor_model_parallel_rank()
            steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]

            head_start = self.num_heads * tp_rank
            head_end = head_start + self.num_heads
            
            for i, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                if head_idx >= head_start and head_idx < head_end:
                    hyperplane, vector = steering_vector
                    hyperplane = hyperplane.to(attn_output.dtype).to(attn_output.device)
                    vector = vector.to(attn_output.dtype).to(attn_output.device)

                    steering_flag = steering_flag.to(attn_output.device)
                    head_offset = head_idx - head_start
                    dim_start = head_offset * self.head_dim
                    dim_end = dim_start + self.head_dim

                    awareness = attn_output[:,dim_start:dim_end] @ hyperplane.T
                    awareness = awareness.sigmoid()
                    awareness = 1 - awareness

                    steering_mask = steering_flag.unsqueeze(-1).float().to(output.dtype)  # (batch_size, 1)
                    awareness = awareness * steering_mask

                    add_vector = vector.repeat(awareness.shape[0], 1) 
                    add_vector = add_vector * awareness

                    attn_output[:, dim_start:dim_end] += self.steering_coef * add_vector
        
    output, _ = self.o_proj(attn_output)

    if self.steering_mode == "after_o_proj":
        if steering_flag.sum() > 0 and \
            layer_idx in self.valid_hyperplane_vector_dict.keys():
                steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]

                for i, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                    hyperplane, vector = steering_vector # 0-> non-linear, 1-> linear 
                    hyperplane = hyperplane.to(output.dtype).to(output.device)
                    vector = vector.to(output.dtype).to(output.device)

                    steering_flag = steering_flag.to(output.device)
                    awareness = output[steering_flag] @ hyperplane.T
                    awareness = awareness.sigmoid() # (k, 1) # 0->non-linear, 1->linear
                    awareness = 1 - awareness # maintain linear part

                    add_vector = vector.repeat(awareness.shape[0], 1) 
                    add_vector = add_vector * awareness

                    output[steering_flag] += self.steering_coef * add_vector
    
    if self.steering_mode == "after_o_proj_no_awareness_norm":
        # cuda graph try need to coment
        # if steering_flag.shape[0] < 256 and \
        if steering_flag is not None and steering_flag.shape[0] > 0 and steering_flag.sum() > 0 and \
            layer_idx in self.valid_hyperplane_vector_dict.keys():
            steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]
            for _, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                hyperplane, vector = steering_vector # 0-> non-linear, 1-> linear 
                vector = vector.to(output.dtype).to(output.device)

                steering_flag = steering_flag.to(output.device)
                # no cuda graph try
                add_vector = vector.repeat(output[steering_flag].shape[0], 1) 
                coeff = torch.tensordot(output[steering_flag], vector, dims=([-1], [-1]))
                coeff = coeff.unsqueeze(-1)
                
                norm_preserve = torch.norm(output[steering_flag], dim=-1)
                
                output[steering_flag] += self.steering_coef * coeff * add_vector

                norm_after = torch.norm(output[steering_flag], dim=-1)

                output[steering_flag] = output[steering_flag] * (norm_preserve / norm_after).unsqueeze(-1)

                # cuda graph try
                # steering_mask = steering_flag.float().to(output.dtype).unsqueeze(-1)  # [batch, 1]
                
                # coeff = torch.sum(output * vector.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
                
                # steering_update = self.steering_coef * coeff * vector.unsqueeze(0)
                
                # norm_preserve = torch.norm(output, dim=-1, keepdim=True)
                
                # output = output + steering_mask * steering_update

                # norm_after = torch.norm(output, dim=-1, keepdim=True)
                
                # norm_ratio = 1.0 + steering_mask * (norm_preserve / norm_after - 1.0)
                # output = output * norm_ratio

    if self.steering_mode == "after_o_proj_norm":
        # cuda graph try need to coment
        # if steering_flag.shape[0] < 256 and \
        if steering_flag is not None and steering_flag.shape[0] > 0 and steering_flag.sum() > 0 and \
        layer_idx in self.valid_hyperplane_vector_dict.keys():
            steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]
            for _, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                hyperplane, vector = steering_vector # 0-> non-linear, 1-> linear 
                vector = vector.to(output.dtype).to(output.device)
                hyperplane = hyperplane.to(output.dtype).to(output.device)

                steering_flag = steering_flag.to(output.device)
                awareness = output[steering_flag] @ hyperplane.T
                awareness = awareness.sigmoid() # (k, 1) # 0->non-linear, 1->linear
                awareness = 1 - awareness # maintain linear part
                
                # no cuda graph try
                add_vector = vector.repeat(output[steering_flag].shape[0], 1) 
                coeff = torch.tensordot(output[steering_flag], vector, dims=([-1], [-1]))
                
                coeff = coeff.unsqueeze(-1)
                
                norm_preserve = torch.norm(output[steering_flag], dim=-1)
                
                output[steering_flag] += self.steering_coef * awareness * coeff * add_vector
                
                norm_after = torch.norm(output[steering_flag], dim=-1)
    
                output[steering_flag] = output[steering_flag] * (norm_preserve / norm_after).unsqueeze(-1)

                # cuda graph try
                # steering_mask = steering_flag.float().to(output.dtype).unsqueeze(-1)  # [batch, 1]
                
                # coeff = torch.sum(output * vector.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
                
                # steering_update = self.steering_coef * coeff * vector.unsqueeze(0)
                
                # norm_preserve = torch.norm(output, dim=-1, keepdim=True)
                
                # output = output + steering_mask * steering_update

                # norm_after = torch.norm(output, dim=-1, keepdim=True)
                
                # norm_ratio = 1.0 + steering_mask * (norm_preserve / norm_after - 1.0)
                # output = output * norm_ratio

    if self.steering_mode == "after_o_proj_norm_threshold":
        # cuda graph try need to coment
        # if steering_flag.shape[0] < 256 and \
        if steering_flag is not None and steering_flag.shape[0] > 0 and steering_flag.sum() > 0 and \
        layer_idx in self.valid_hyperplane_vector_dict.keys():
            steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]
            for _, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                hyperplane, vector = steering_vector # 0-> non-linear, 1-> linear 
                vector = vector.to(output.dtype).to(output.device)
                hyperplane = hyperplane.to(output.dtype).to(output.device)

                steering_flag = steering_flag.to(output.device)
                awareness = output[steering_flag] @ hyperplane.T
                awareness = awareness.sigmoid() # (k, 1) # 0->non-linear, 1->linear
                awareness = 1 - awareness # maintain linear part
                awareness = (awareness > 0.5).float()
                
                # no cuda graph try
                add_vector = vector.repeat(output[steering_flag].shape[0], 1) 
                coeff = torch.tensordot(output[steering_flag], vector, dims=([-1], [-1]))
                
                coeff = coeff.unsqueeze(-1)
                
                norm_preserve = torch.norm(output[steering_flag], dim=-1)
                
                output[steering_flag] += self.steering_coef * awareness * coeff * add_vector
                
                norm_after = torch.norm(output[steering_flag], dim=-1)

                output[steering_flag] = output[steering_flag] * (norm_preserve / norm_after).unsqueeze(-1)

                # cuda graph try
                # steering_mask = steering_flag.float().to(output.dtype).unsqueeze(-1)  # [batch, 1]
                
                # coeff = torch.sum(output * vector.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
                
                # steering_update = self.steering_coef * coeff * vector.unsqueeze(0)
                
                # norm_preserve = torch.norm(output, dim=-1, keepdim=True)
                
                # output = output + steering_mask * steering_update

                # norm_after = torch.norm(output, dim=-1, keepdim=True)
                
                # norm_ratio = 1.0 + steering_mask * (norm_preserve / norm_after - 1.0)
                # output = output * norm_ratio

    if self.steering_mode == "after_o_proj_no_awareness_norm":
        # cuda graph try need to coment
        # if steering_flag.shape[0] < 256 and \
        if steering_flag is not None and steering_flag.shape[0] > 0 and steering_flag.sum() > 0 and \
            layer_idx in self.valid_hyperplane_vector_dict.keys():
            steering_vector_dict = self.valid_hyperplane_vector_dict[layer_idx]
            for _, (head_idx, steering_vector) in enumerate(steering_vector_dict.items()):
                hyperplane, vector = steering_vector # 0-> non-linear, 1-> linear 
                vector = vector.to(output.dtype).to(output.device)

                steering_flag = steering_flag.to(output.device)
                # no cuda graph try
                add_vector = vector.repeat(output[steering_flag].shape[0], 1) 
                coeff = torch.tensordot(output[steering_flag], vector, dims=([-1], [-1]))
                coeff = coeff.unsqueeze(-1)
                
                norm_preserve = torch.norm(output[steering_flag], dim=-1)
                
                output[steering_flag] += self.steering_coef * coeff * add_vector

                norm_after = torch.norm(output[steering_flag], dim=-1)

                output[steering_flag] = output[steering_flag] * (norm_preserve / norm_after).unsqueeze(-1)

                # cuda graph try
                # steering_mask = steering_flag.float().to(output.dtype).unsqueeze(-1)  # [batch, 1]
                
                # coeff = torch.sum(output * vector.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
                
                # steering_update = self.steering_coef * coeff * vector.unsqueeze(0)
                
                # norm_preserve = torch.norm(output, dim=-1, keepdim=True)
                
                # output = output + steering_mask * steering_update

                # norm_after = torch.norm(output, dim=-1, keepdim=True)
                
                # norm_ratio = 1.0 + steering_mask * (norm_preserve / norm_after - 1.0)
                # output = output * norm_ratio
                
    return output + hidden_states


def steering_layer_forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int=None,
    steering_flag: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    attn_output = self.attn(hidden_states, positions, layer_idx=layer_idx, steering_flag=steering_flag)
    output = self.mlp(attn_output)
    return output

def steering_model_forward(self, input_ids: torch.Tensor,
            positions: torch.Tensor, steering_flag: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = self.embedding(input_ids)
    for i, layer in enumerate(self.layers):
        x = layer(x, positions, i, steering_flag)
    x = self.norm(x)
    return x

def set_steering_flag(self, steering_flag):
    num_layers = len(self.model.layers)
    for i in range(num_layers):
        self.model.layers[i].self_attn.steering_flag = steering_flag

def causal_lm_forward(self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors] = None,
    inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    from vllm.forward_context import get_forward_context
    context = get_forward_context()

    steering_flag = None    
    if self.if_steering and context is not None and context.attn_metadata is not None:
        # print("===========", context.attn_metadata.keys())
        if context.attn_metadata['model.block.1.attn.attn'].max_query_len > 1:
            # prefill
            self.new_round = True
        else:
            # decode
            if self.new_round:
                # first time enter decode after prefill
                self.new_round = False
                # self.steering_think_flag = (input_ids == self.steering_think_start_id).to(torch.bool)
            
            # self.steering_think_flag = torch.logical_or(self.steering_think_flag, input_ids == self.steering_think_start_id)
            # self.steering_think_flag = torch.logical_and(self.steering_think_flag, input_ids != self.steering_think_end_id)
            split_flag = torch.isin(input_ids, self.steering_split_ids.to(input_ids.device))
            # steering_flag = torch.logical_and(split_flag, self.steering_think_flag)
            steering_flag = split_flag

    return self.model(input_ids, positions, steering_flag)

def load_steering_vector(steering_vector_path, steering_number, device):
    steering_vector_dict = torch.load(os.path.join(steering_vector_path, "probe_best.pt"))
    layer_num = len(steering_vector_dict.keys())
    head_num = len(steering_vector_dict[list(steering_vector_dict.keys())[0]].keys())

    # create 2D accuracy matrix for each layer, head
    accuracy_matrix = np.zeros((layer_num, head_num))
    for layer in steering_vector_dict.keys():
        for head in steering_vector_dict[layer].keys():
            accuracy_matrix[layer, head] = steering_vector_dict[layer][head]['accuracy']

    layer_head_idx = topk_indices(accuracy_matrix, steering_number)

    valid_hyperplane_vector_dict = {}
    for acc, layer, head in layer_head_idx:
        layer = int(layer)
        head = int(head)
        hyperplane = steering_vector_dict[layer][head]['model_dict']['weight']
        vector = steering_vector_dict[layer][head]['steering_vector']
        if layer not in valid_hyperplane_vector_dict.keys():
            valid_hyperplane_vector_dict[layer] = {head: (hyperplane.to(device), vector.to(device))}
        else:
            valid_hyperplane_vector_dict[layer][head] = (hyperplane.to(device), vector.to(device))

    return valid_hyperplane_vector_dict

def monkey_patch_gpt_oss_vllm(steering_vector_path, steering_number, steering_coef, steering_mode, model_name_or_path):

    device = torch.device("cuda")

    steering_number = int(steering_number)
    steering_coef = float(steering_coef)

    # patch the forward function of the model
    print('Enabling steering... for path: ', steering_vector_path)

    valid_hyperplane_vector_dict = load_steering_vector(steering_vector_path, steering_number, device)

    OAIAttention.valid_hyperplane_vector_dict = valid_hyperplane_vector_dict
    OAIAttention.steering_coef = steering_coef
    OAIAttention.steering_mode = steering_mode
    
    OAIAttention.forward = steering_attention_forward

    TransformerBlock.forward = steering_layer_forward

    TransformerBlock.steering_flag = None
    GptOssModel.forward = steering_model_forward

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    vocab = tokenizer.get_vocab()

    GptOssForCausalLM.new_round = False
    GptOssForCausalLM.steering_think_flag = None

    GptOssForCausalLM.steering_split_ids = torch.LongTensor([vocab[token] for token in vocab.keys() if "ĊĊ" in token]).to(device)
    
    GptOssForCausalLM.logit_adjustment_tokens = None
    GptOssForCausalLM.logit_adjustment_values = None
    GptOssForCausalLM.logit_adjustment_position = None
    GptOssForCausalLM.logit_adjustment_max_len = None

    GptOssForCausalLM.if_steering = True

    GptOssForCausalLM.set_steering_flag = set_steering_flag
    GptOssForCausalLM.forward = causal_lm_forward

    print("Steering vector enabled!")