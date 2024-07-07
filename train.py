import sys
from typing import Callable

import numpy as np
import torch


def generate_text(
    model: torch.nn.Module,
    start_prompt: str,
    max_tokens: int,
    temp: float,
    vocab: dict[str, int],
    device: torch.device,
) -> str:
    """Function to Generate Text During Training."""

    def sample_from(probs: np.ndarray, temp: float) -> int:
        """Sample a token from the list of probabilities."""
        probs = probs ** (1 / temp)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    with torch.no_grad():
        vocab_words = list(vocab.keys())
        # If not found return the unknown token (1)
        start_tokens = [vocab.get(x, 1) for x in start_prompt.split()]
        sample_token = None
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = torch.LongTensor([start_tokens]).to(device)
            y = model(x)
            sample_token = sample_from(
                torch.nn.functional.softmax(y[0][-1], dim=0).cpu().numpy(),
                temp,
            )
            start_tokens.append(sample_token)
            start_prompt += f" {vocab_words[sample_token]}"
        return f"\ngenerated text:\n{start_prompt}\n"


def train_network(
    model: torch.nn.Module,
    vocab: dict[str, int],
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Train the Network."""
    print("Training Started")
    sys.stdout.flush()
    for epoch in range(1, num_epochs + 1):
        train_loss = []
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            outputs = model(x)
            loss = loss_function(torch.transpose(outputs, 2, 1), y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        print(f"Epoch: {epoch}, Training Loss: {np.mean(train_loss):.4f}")
        generate_text(
            model,
            "wine review",
            max_tokens=80,
            temp=1.0,
            vocab=vocab,
            device=device,
        )
