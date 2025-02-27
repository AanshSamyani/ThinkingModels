import torch as t
import nnsight
from tqdm import tqdm
import os
from nnsight import LanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def save_text_to_file(text: dict,
                      filename:str,
                      dir_path: str):
    """
    Save the given text to a file.

    Args:
        text (dict): The text to save.
        filename (str): The name of the file.
        dir_path (str): The directory path where the file will be saved.
    """
    os.makedirs(dir_path, exist_ok=True)  
    file_path = os.path.join(dir_path, filename)  
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"File saved at: {file_path}")
    
def save_acts_to_file(hidden_states: dict,
                      filename: str,
                      dir_path: str):

    """
    Saves the given hidden states dictionary to a file.

    Parameters:
    - hidden_states (dict): Dictionary containing hidden states or activations.
    - filename (str): Name of the file to save the data.
    - dir_path (str): Directory where the file should be saved.

    The function ensures that the specified directory exists, then saves the 
    hidden states using torch's save function and prints the file path.
    """
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
    
    """
    Generates output from a language model given a prompt and optionally 
    saves the activations and the generated answer.

    Parameters:
    - llm (LanguageModel): The language model instance.
    - prompt (str): Input prompt for generation.
    - append_eos_token (bool, optional): If True, appends the EOS token to the prompt. Default is True.
    - n_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 5000.
    - temperature (float, optional): Sampling temperature for generation. Default is 1.0.
    - print_ans (bool, optional): If True, prints the prompt and generated answer. Default is True.
    - save_ans (bool, optional): If True, saves the generated answer to a file. Default is False.
    - ans_filename (str, optional): Filename to save the answer. Default is "ans.txt".
    - ans_dir_path (str, optional): Directory path for saving the answer. Default is "/home/ans_dir_path".
    - save_activations (bool, optional): If True, saves the model activations. Default is False.
    - act_filename (str, optional): Filename to save the activations. Default is "acts.pth".
    - act_dir_path (str, optional): Directory path for saving activations. Default is "/home/act_dir_path".

    The function:
    1. Appends an EOS token to the prompt if required.
    2. Generates output using the model while tracking activations.
    3. Stores activations for each layer, including MLP and attention projections.
    4. Prints the decoded prompt and generated response.
    5. Saves the generated answer if `save_ans` is True.
    6. Saves the activations if `save_activations` is True.
    """

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
        
    