from collections import Counter
import numpy as np
import torch
import os
import pickle
import random
from torch import nn
from tqdm import tqdm
from transformers import LlamaForCausalLM

from model.configuration_llama_moe import LlamaMoEConfig
from model.modeling_llama_moe import LlamaMoEForCausalLM


def random_split(config):
    neuron_total = config.intermediate_size
    num_neurons = neuron_total // config.num_experts
    neuron_labels = np.arange(0, config.num_experts, dtype=int).tolist()
    neuron_labels = neuron_labels * num_neurons
    random.shuffle(neuron_labels)
    indices = torch.tensor(neuron_labels, dtype=torch.int)
    experts_neurons = [torch.where(indices==j)[0].tolist() for j in range(config.num_experts)]
    return neuron_labels, experts_neurons


def convert_to_llama_moe(
        config: LlamaMoEConfig,
        llama_model_path,
        save_path,
        score_scale_factor=None,
):
    """
    LlamaMoEForCausalLM
    """

    moe_neuron_indices = []

    """load model"""
    print("Loading llama model...")
    dtype = config.torch_dtype
    model_llama = LlamaForCausalLM.from_pretrained(llama_model_path,
                                                   torch_dtype=dtype)
    model_llama.to("cuda")
    model_llama_state_dict = model_llama.state_dict()


    """generate indices using random_split"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for _ in tqdm(range(num_layers), desc="generating neuron indices"):
        _, experts_neurons = random_split(config)
        moe_neuron_indices.append(experts_neurons)

    """build config"""
    print("Buiding llama-moe config...")
    config.score_scale_factor = (1.0 if score_scale_factor is None else score_scale_factor)

    """initialize moe model"""
    print("Initializing llama-moe model...")
    config.moe_intermediate_size = config.intermediate_size // config.num_experts
    model_llama_moe = LlamaMoEForCausalLM(config)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()
    print(model_llama_moe.state_dict)
    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in tqdm(model_llama_state_dict.keys(), desc="converting weights"):
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().to(dtype)
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(config.num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.experts.{}.gate_proj.weight".format(layer_index, expert_index)] = \
                        model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().to(dtype)
                elif "up" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.experts.{}.up_proj.weight".format(layer_index, expert_index)] = \
                        model_llama_state_dict[key][moe_neuron_indices[layer_index][expert_index]].cpu().to(dtype)
                elif "down" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.experts.{}.down_proj.weight".format(layer_index, expert_index)] = \
                        model_llama_state_dict[key].transpose(0, 1)[moe_neuron_indices[layer_index][expert_index]].transpose(0, 1).cpu().to(dtype)

    for layer_index in tqdm(range(num_layers), desc="constructing routers "):
        ll = nn.Linear(config.hidden_size, config.num_experts, bias=config.router_bias)
        model_llama_moe_state_dict["model.layers.{}.mlp.gate.ll.weight".format(layer_index)] = ll.state_dict()['weight'].cpu().to(dtype)
        if config.router_bias:
            model_llama_moe_state_dict["model.layers.{}.mlp.gate.ll.bias".format(layer_index)] = ll.state_dict()['bias'].cpu().to(dtype)
    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.to(dtype)

    print("Saving converted model...")
    config.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEForCausalLM saved to "{save_path}".')
