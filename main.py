from pathlib import Path
from model import GPTModel
import torch
from dataset import WineDataset
import pytorch_model_summary as pms

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20

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

model = GPTModel(max_len_input=MAX_LEN, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, feed_forward_dim=FEED_FORWARD_DIM, num_heads=N_HEADS, key_dim=KEY_DIM)
model.to(device)
pms.summary(
    model,
    torch.zeros((BATCH_SIZE, MAX_LEN)).to(device=device, dtype=torch.int),
    show_input=False,
    print_summary=True,
    max_depth=5,
    show_parent_layers=True,
)

