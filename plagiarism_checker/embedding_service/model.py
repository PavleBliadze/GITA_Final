from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "microsoft/codebert-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_embedding(code: str):
    tokens = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}  # move to GPU if available
    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding.tolist()