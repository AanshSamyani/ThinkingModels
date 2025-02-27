import torch as t
import nnsight
from tqdm import tqdm
from nnsight import LanguageModel
from safetensors.torch import load_file
import os


def rearrange_activations_mlp(hidden_states: dict):
    ls_activations = []
    for i in range(len(hidden_states)):
        first_token_acts = hidden_states[f"layer_{i}"]["mlp"][0]
        rest_token_acts = t.cat(hidden_states[f"layer_{i}"]["mlp"][1:], dim=1)
        all_acts = t.cat([first_token_acts, rest_token_acts], dim=1)
        ls_activations.append(all_acts)
    
    return t.cat(ls_activations, dim=0)

def get_think_token_position(llm: LanguageModel, 
                             ans_dir_path: str,
                             ans_filename: str) -> int:
    
    ans_file_path = os.path.join(ans_dir_path, ans_filename)
    with open(ans_file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()
    tokenized_text_ids = llm.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokenized_text = llm.tokenizer.tokenize(text)
    think_token = "</think>"
    think_token_index = tokenized_text.index(think_token)
    return tokenized_text_ids, tokenized_text, think_token_index     

def get_think_token_mlp_activations(act_dir_path: str,
                                    act_filename: str) -> t.Tensor:
    
    act_file_path = os.path.join(act_dir_path, act_filename)
    hidden_states = t.load(act_file_path)
    return rearrange_activations_mlp(hidden_states)