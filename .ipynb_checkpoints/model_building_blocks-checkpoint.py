from typing import Callable

import torch
import torch.nn.functional as F


def create_attention_mask(
    key_length: int,
    query_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create a Casual Mask for
    the multi head attention layer.
    """
    i = torch.arange(query_length)[:, None]
    j = torch.arange(key_length)
    # Create a mask of size (query_length, key_length)
    # (i, j) is true if i >= j - key_length + query_length
    mask = i >= j - key_length + query_length
    # Cast the mask to the dtype
    mask = torch.logical_not(mask)
    mask = mask.to(dtype)
    return mask


class TokenAndPositionEmbedding(torch.nn.Module):
    """Token and positioning embedding layer for a sequence."""

    def __init__(
        self, max_len_input: int, vocab_size: int, embed_dim: int
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.token_emb = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.position_emb = torch.nn.Embedding(
            num_embeddings=max_len_input, embedding_dim=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        len_input = x.size()[1]
        positions = torch.arange(start=0, end=len_input, step=1).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        position_embedding = self.position_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding


class TransformerBlock(torch.nn.Module):
    """Transformer Block Layer."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        embed_dim: int,
        ff_dim: int,
        mask_function: Callable[[int, int, torch.dtype], torch.Tensor],
        dropout_rate: float = 0.1,
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            batch_first=True,
        )
        self.dropout_1 = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=embed_dim, eps=1e-6
        )
        self.ffn_1 = torch.nn.Linear(
            in_features=embed_dim, out_features=ff_dim
        )
        self.ffn_2 = torch.nn.Linear(
            in_features=ff_dim, out_features=embed_dim
        )
        self.dropout_2 = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=embed_dim, eps=1e-6
        )
        self.mask_function = mask_function

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        seq_len = inputs.size()[1]
        mask = self.mask_function(seq_len, seq_len, torch.bool).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        attention_output, _ = self.attn(
            query=inputs, key=inputs, value=inputs, attn_mask=mask
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.layer_norm_1(inputs + attention_output)
        ffn_1 = F.relu(self.ffn_1(out1))
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        output = self.layer_norm_2(out1 + ffn_output)
        return output
