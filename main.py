from pathlib import Path

import pytorch_model_summary as pms
import torch
import os

from dataset import WineDataset
from model import GPTModel
from train import generate_text, train_network

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
BATCH_SIZE = 128
EPOCHS = 100 # Initially 300

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wine_dataset = WineDataset(
    path_to_file=Path("data/wine-reviews/winemag-data-130k-v2.json"),
    max_length=MAX_LEN,
    vocab_size=VOCAB_SIZE,
)
vocab = wine_dataset.get_vocab()

trainloader = torch.utils.data.DataLoader(
    wine_dataset, batch_size=BATCH_SIZE, shuffle=True
)

model = GPTModel(
    max_len_input=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    feed_forward_dim=FEED_FORWARD_DIM,
    num_heads=N_HEADS,
    key_dim=KEY_DIM,
)
model.to(device)
pms.summary(
    model,
    torch.zeros((BATCH_SIZE, MAX_LEN)).to(device=device, dtype=torch.int),
    show_input=False,
    print_summary=True,
    max_depth=5,
    show_parent_layers=True,
)

optimizer = torch.optim.Adam(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=0.001,
)

train_network(
    model=model,
    vocab=vocab,
    num_epochs=EPOCHS,
    optimizer=optimizer,
    loss_function=torch.nn.CrossEntropyLoss(),
    trainloader=trainloader,
    device=device,
)

def save_model_with_fallback(model, filename='name.pth'):
    """
    Save model with fallback locations for different environments
    """
    save_paths = [
        filename,  # Current directory (works in local environments)
        f'/kaggle/working/{filename}',  # Kaggle notebooks
        f'/tmp/{filename}',  # Temporary directory fallback
    ]
    
    for path in save_paths:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(model, path)
            print(f"Model saved successfully to '{path}'")
            return path
        except (RuntimeError, OSError, PermissionError) as e:
            print(f"Failed to save to '{path}': {e}")
            continue
    
    raise RuntimeError("Could not save model to any location")

# Usage
try:
    saved_path = save_model_with_fallback(model, 'name.pth')
except RuntimeError as e:
    print(f"Error: {e}")

print(
    generate_text(
        model,
        "wine review : us",
        max_tokens=80,
        temp=1.0,
        vocab=vocab,
        device=device,
    )
)

print(
    generate_text(
        model,
        "wine review : italy",
        max_tokens=80,
        temp=0.5,
        vocab=vocab,
        device=device,
    )
)

print(
    generate_text(
        model,
        "wine review : germany",
        max_tokens=80,
        temp=0.5,
        vocab=vocab,
        device=device,
    )
)
