import torch as t
import nnsight
from tqdm import tqdm
from nnsight import LanguageModel
from safetensors.torch import load_file
import os


def rearrange_activations_mlp(hidden_states: dict):
    """
    Rearranges MLP activations from hidden states to a structured tensor.

    Parameters:
    - hidden_states (dict): A dictionary containing activations for each layer.

    Returns:
    - t.Tensor: A concatenated tensor containing MLP activations across all layers.

    The function:
    1. Iterates over all layers in `hidden_states`.
    2. Extracts the activations of the first token separately.
    3. Concatenates the remaining tokens' activations along the second dimension.
    4. Collects activations for all layers and returns a single concatenated tensor.
    """
    
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
    """
    Finds the position of the `</think>` token in a generated answer.

    Parameters:
    - llm (LanguageModel): The language model used for tokenization.
    - ans_dir_path (str): Directory path where the answer file is stored.
    - ans_filename (str): Name of the file containing the generated answer.

    Returns:
    - tuple: (tokenized_text_ids, tokenized_text, think_token_index)
        - tokenized_text_ids (torch.Tensor): Tokenized representation of the text.
        - tokenized_text (list of str): List of tokenized words.
        - think_token_index (int): Index of the `</think>` token in the tokenized text.

    The function:
    1. Reads the generated answer from a file.
    2. Tokenizes the text using the language model’s tokenizer.
    3. Identifies the position of the `</think>` token and returns it.
    """
    
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
    
    """
    Loads and rearranges MLP activations associated with the `</think>` token.

    Parameters:
    - act_dir_path (str): Directory path where activation files are stored.
    - act_filename (str): Name of the file containing saved activations.

    Returns:
    - t.Tensor: A tensor containing rearranged MLP activations.

    The function:
    1. Loads the saved hidden states from a file.
    2. Calls `rearrange_activations_mlp()` to process and structure the activations.
    3. Returns the structured activation tensor.
    """
    
    act_file_path = os.path.join(act_dir_path, act_filename)
    hidden_states = t.load(act_file_path)
    return rearrange_activations_mlp(hidden_states)