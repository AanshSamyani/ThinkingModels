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
    """
    Saves a given text string to a file in the specified directory.

    Parameters:
    - text (str): The text content to be saved.
    - filename (str): The name of the file to save the text in.
    - dir_path (str): The directory where the file should be saved.

    The function:
    1. Ensures the target directory exists.
    2. Constructs the full file path.
    3. Writes the text to the specified file.
    4. Prints a message confirming the file's location.
    """
    
    os.makedirs(dir_path, exist_ok=True)  
    file_path = os.path.join(dir_path, filename)  
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"File saved at: {file_path}")

def load_sae(layer: int):
    """
    Loads a Sparse AutoEncoder (SAE) model for a given transformer layer.

    Parameters:
    - layer (int): The index of the transformer layer to load the SAE for.

    Returns:
    - AutoEncoder: The loaded SAE model with pre-trained weights.

    The function:
    1. Defines the expected weight file path.
    2. Initializes the SAE with a fixed activation and dictionary size.
    3. Loads the state dictionary from a safetensors file.
    4. Adjusts weight naming conventions for compatibility.
    5. Transfers the model to GPU and returns it.
    """
    
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
    """
    Extracts sparse autoencoder (SAE) features from MLP activations.

    Parameters:
    - mlp_acts (t.Tensor): The tensor containing MLP activations.
    - sae (AutoEncoder): The trained SAE model for feature extraction.
    - layer (int, optional): The transformer layer from which to extract features (default: 0).

    Returns:
    - t.Tensor: The extracted SAE features.

    The function:
    1. Uses a no-gradient context to prevent backpropagation.
    2. Encodes the activations using the SAE model.
    3. Returns the extracted sparse features.
    """
    
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
    """
    Guides the LLM's output using modified latent representations from an SAE.

    Parameters:
    - llm (LanguageModel): The language model generating the response.
    - prompt (str): The input prompt to start text generation.
    - append_eos_token (bool, optional): Whether to append EOS token to the prompt (default: True).
    - n_new_tokens (int, optional): The number of tokens to generate (default: 200).
    - temperature (float, optional): Sampling temperature for generation (default: 1).
    - print_ans (bool, optional): Whether to print the generated output (default: False).
    - save_ans (bool, optional): Whether to save the generated output (default: False).
    - ans_filename (str, optional): Filename for saving the answer (default: "ans.txt").
    - ans_dir_path (str, optional): Directory path for saving the answer (default: "/home/ans_dir_path").
    - save_activations (bool, optional): Whether to save model activations (default: False).
    - act_filename (str, optional): Filename for saving activations (default: "acts.pth").
    - act_dir_path (str, optional): Directory path for saving activations (default: "/home/act_dir_path").
    - sae (AutoEncoder, optional): The SAE model to modify latent representations (default: None).
    - layer (int, optional): The transformer layer to modify (default: 10).
    - top_think_token_latents_indices (list, optional): Indices of top activations to modify (default: None).
    - top_k_indices (int, optional): Number of top activations to modify (default: 5).
    - scaling_factor (int, optional): Scaling factor to amplify top activations (default: 5).

    The function:
    1. Appends an EOS token to the prompt if required.
    2. Generates text token-by-token, modifying MLP outputs for the specified layer.
    3. Extracts activations, encodes them into latent space, scales top activations, and decodes them back.
    4. Injects the modified latent representation into the model.
    5. Saves or prints the generated response if specified.
    """
    
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