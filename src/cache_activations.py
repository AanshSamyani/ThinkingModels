import torch as t
import nnsight
from tqdm import tqdm
import os
from nnsight import LanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def save_text_to_file(text: dict,
                      filename:str,
                      dir_path: str):
    
    os.makedirs(dir_path, exist_ok=True)  
    file_path = os.path.join(dir_path, filename)  
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"File saved at: {file_path}")
    
def save_acts_to_file(hidden_states: dict,
                      filename: str,
                      dir_path: str):

    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    t.save(hidden_states, file_path)
    print(f"File saved at: {file_path}")
    
def generate_output(llm: LanguageModel,
                    prompt:str,
                    append_eos_token: bool = True,
                    n_new_tokens:int = 5000,
                    temperature: float = 1,
                    print_ans: bool = True,
                    save_ans: bool = False, 
                    ans_filename: str = "ans.txt",
                    ans_dir_path: str = "/home/ans_dir_path",
                    save_activations: bool = False,
                    act_filename: str = "acts.pth",
                    act_dir_path: str = "/home/act_dir_path"):
    
    if append_eos_token:
        prompt += llm.tokenizer.eos_token
    
    with llm.generate(prompt, max_new_tokens=n_new_tokens, temperature=temperature) as tracer:
        hidden_states = nnsight.dict().save()
        for i in range(len(llm.model.layers)):
            hidden_states[f"layer_{i}"] = {
                "mlp": [],
                "q_proj": [],
                "k_proj": [],
            }
        llm.model.layers.all()
        for i in range(len(llm.model.layers)):
            hidden_states[f"layer_{i}"]["mlp"].append(llm.model.layers[i].mlp.output.cpu())
            hidden_states[f"layer_{i}"]["q_proj"].append(llm.model.layers[i].self_attn.q_proj.output.cpu())
            hidden_states[f"layer_{i}"]["k_proj"].append(llm.model.layers[i].self_attn.k_proj.output.cpu())
        out = llm.generator.output.cpu().save()
        
    decoded_prompt = llm.tokenizer.decode(out[0][0:50])
    decoded_answer = llm.tokenizer.decode(out[0][:])
    
    if print_ans:
        print(f"Prompt: {decoded_prompt}")
        print(f"Generated Answer: {decoded_answer}")
        
    if save_ans:
        save_text_to_file(decoded_answer, ans_filename, ans_dir_path)
        
    if save_activations:
        save_acts_to_file(hidden_states, act_filename, act_dir_path)
        
    