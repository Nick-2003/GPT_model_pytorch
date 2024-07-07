from pathlib import Path

import torch

from dataset import WineDataset

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20


wine_dataset = WineDataset(
    path_to_file=Path("data/wine-reviews/winemag-data-130k-v2.json"),
    max_length=MAX_LEN,
    vocab_size=VOCAB_SIZE,
)
vocab = wine_dataset.get_vocab()

trainloader = torch.utils.data.DataLoader(
    wine_dataset, batch_size=BATCH_SIZE, shuffle=True
)
