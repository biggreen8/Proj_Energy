import tqdm
import torch
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from probs import n_digits

# first, create an access token for Gemma and then replace the blank inside the quotations in the line below.  Then uncomment the line below
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=access_token
)

import os

# Detect the device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple M1/M2 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif 'HSA_PATH' in os.environ or os.path.exists('/opt/rocm'):
    device = torch.device("cuda")  # Assume ROCm is available (PyTorch treats this as CUDA)
else:
    device = torch.device("cpu")   # Fallback to CPU

# Example usage
print(f"Using device: {device}")



class Toks(Dataset):
    def __init__(self, toks):
        self.toks = toks
    
    def __len__(self):
        return len(self.toks["input_ids"])
    
    def __getitem__(self, idx):
        return self.toks["input_ids"][idx], self.toks["attention_mask"][idx]


for i in range(1,n_digits+1):
    # Create separate folders for "n_digits by i" and "i by n_digits"
    n_by_i= f"{n_digits}_by_{i}_results"
    i_by_n = f"{i}_by_{n_digits}_results"
    
    # Create the directories if they don't exist
    os.makedirs(n_by_i, exist_ok=True)
    os.makedirs(i_by_n, exist_ok=True)
    
    examples = open(f"{n_digits}_by_{i}_problems.txt", "r").readlines()
    examples1 = open(f"{i}_by_{n_digits}_problems.txt", "r").readlines()

    few_shot_examples = "100 + 200 = 300\n520 + 890 = 1410\n"

    a = [few_shot_examples + v.strip() + " " for v in examples]
    toked = tokenizer.batch_encode_plus(a, return_tensors="pt", padding=True)

    b = [few_shot_examples + s.strip() + " " for s in examples1]
    toked1 = tokenizer.batch_encode_plus(b, return_tensors="pt", padding=True)


    dl = DataLoader(Toks(toked), batch_size=32)
    dl1 = DataLoader(Toks(toked1), batch_size=32)

    # Loop over values of 'u' for temperature settings
    for u in np.arange(0, 2.1, 0.1):  # 2.1 to include the value 2.0
        texts = []  # Reset texts for each 'u'
        for x, y in tqdm.tqdm(dl, desc=f"Processing u={u:.1f}"):
            x = x.to(device)
            y = y.to(device)
            outputs = model.generate(input_ids=x, attention_mask=y, max_new_tokens=32, temperature=u)
            texts.append(tokenizer.batch_decode(outputs))

        # Save the results to a .pkl file inside the "n_digits by i" folder for each 'u'
        with open(f"{n_by_i}/{n_digits}_by_{i}_at_{u:.1f}_results.pkl", "wb") as f:
            pickle.dump(texts, f)

    # Loop over values of 't' for temperature settings
    for t in np.arange(0, 2.1, 0.1):  # 2.1 to include the value 2.0
        texts1 = []  # Reset texts1 for each 't'
        for f, g in tqdm.tqdm(dl1, desc=f"Processing t={t:.1f}"):
            f = f.to(device)
            g = g.to(device)
            outputs1 = model.generate(input_ids=f, attention_mask=g, max_new_tokens=32, temperature=t)
            texts1.append(tokenizer.batch_decode(outputs1))

        # Save the results to a .pkl file inside the "i by n_digits" folder for each 't'
        with open(f"{i_by_n}/{i}_by_{n_digits}_at_{t:.1f}_results.pkl", "wb") as f:
            pickle.dump(texts1, f)