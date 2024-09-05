import tqdm
import torch
import pickle
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
    examples = open(f"{n_digits}_{i}_digit_problems.txt", "r").readlines()
    examples1 = open(f"{i}_{n_digits}_digit_problems.txt", "r").readlines()
    few_shot_examples = "100 + 200 = 300\n520 + 890 = 1410\n"
    a = [few_shot_examples + v.strip() + " " for v in examples]
    toked = tokenizer.batch_encode_plus(a, return_tensors="pt", padding=True)
    b = [few_shot_examples + v.strip() + " " for v in examples1]
    toked1 = tokenizer.batch_encode_plus(b, return_tensors="pt", padding=True)


    dl = DataLoader(Toks(toked), batch_size=32)
    dl1 = DataLoader(Toks(toked1), batch_size=32)
    texts = []
    for x, y in tqdm.tqdm(dl):
        x = x.to(device)
        y = y.to(device)
        outputs = model.generate(input_ids=x, attention_mask=y, max_new_tokens=32)
        texts.append(tokenizer.batch_decode(outputs))

    for x, y in tqdm.tqdm(dl1):
        x = x.to(device)
        y = y.to(device)
        outputs = model.generate(input_ids=x, attention_mask=y, max_new_tokens=32)
        texts.append(tokenizer.batch_decode(outputs))

    pickle.dump(texts, open(f"{n_digits}_{i}_digit_results.pkl", "wb"))
    pickle.dump(texts, open(f"{i}_{n_digits}_digit_results.pkl", "wb"))