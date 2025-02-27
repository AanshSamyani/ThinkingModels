import torch as t
import nnsight
from tqdm import tqdm
from nnsight import LanguageModel
from safetensors.torch import load_file
import os
from dictionary_learning.dictionary import AutoEncoder

def save_text_to_file(text: dict,
                      filename:str,
                      dir_path: str):
    
    os.makedirs(dir_path, exist_ok=True)  
    file_path = os.path.join(dir_path, filename)  
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"File saved at: {file_path}")

def load_sae(layer: int):
    weights_path = f"/home/sae_model/layers.{layer}.mlp/sae.safetensors"
    activation_dim =  1536 
    dictionary_size = 65536 
    ae = AutoEncoder(activation_dim, dictionary_size)
    state_dict = load_file(weights_path)
    state_dict.keys()
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key  
        if key == "W_dec":
            new_key = "decoder.weight"
            new_state_dict[new_key] = state_dict[key].T  
        elif key == "b_dec":
            new_key = "bias"
        if key != "W_dec": 
            new_state_dict[new_key] = state_dict[key]  # Copy data to the new key

    ae.load_state_dict(new_state_dict)
    ae.cuda() 
    return ae 


def get_sae_features(mlp_acts: t.Tensor,
                     sae: AutoEncoder,
                     layer: int = 0) -> t.Tensor:
    
    with t.no_grad():
        features = sae.encode(mlp_acts[:, layer, :].cuda())
    return features

def steer_with_sae_latents(llm: LanguageModel,
                           prompt: str, 
                           append_eos_token: bool = True,
                           n_new_tokens: int = 200,
                           temperature: float = 1,
                           print_ans: bool = False,
                           save_ans: bool = False,
                           ans_filename: str = "ans.txt",
                           ans_dir_path: str = "/home/ans_dir_path",
                           save_activations: bool = False,
                           act_filename: str = "acts.pth",
                           act_dir_path: str = "/home/act_dir_path",
                           sae: AutoEncoder = None,
                           layer: int = 10, 
                           top_think_token_latents_indices: list = None,
                           top_k_indices: int = 5,
                           scaling_factor: int = 5):
    
    if append_eos_token:
        prompt += llm.tokenizer.eos_token
        
    with llm.generate(prompt, max_new_tokens=n_new_tokens, temperature=temperature) as tracer:
        for i in range(n_new_tokens):
            if i==0:
                lt_output = llm.model.layers[layer].mlp.output.cpu()
                lt_latents = sae.encode(lt_output[:, -1, :].cuda())
                for j in top_think_token_latents_indices[:top_k_indices]:
                    lt_latents[0][j] = lt_latents[0][j] * scaling_factor
                    
                lt_output = sae.decode(lt_latents)
                llm.model.layers[layer].mlp.output[:, -1, :] = lt_output
                
        out = llm.generator.output.cpu().save()
        
    decoded_answer = llm.tokenizer.decode(out[0][:])
    
    if print_ans:
        print(f"Generated Answer: {decoded_answer}")
        
    if save_ans:
        save_text_to_file(decoded_answer, ans_filename, ans_dir_path)