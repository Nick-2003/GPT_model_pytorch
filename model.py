import torch
from model_building_blocks import TokenAndPositionEmbedding
from model_building_blocks import create_attention_mask
from model_building_blocks import TransformerBlock


class GPTModel(torch.nn.Module):
    """GPT Model Class."""
    def __init__(self, max_len_input: int, vocab_size: int, embed_dim: int, feed_forward_dim: int, num_heads: int, key_dim: int) -> None:
        """Init Function."""
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(max_len_input=max_len_input, vocab_size=vocab_size, embed_dim=embed_dim)
        self.transformer = TransformerBlock(num_heads=num_heads, key_dim=key_dim, embed_dim=embed_dim, ff_dim=feed_forward_dim, mask_function=create_attention_mask)
        self.output_layer = torch.nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        embedding = self.embedding_layer(input_tensor)
        transformer_output = self.transformer(embedding)
        output = self.output_layer(transformer_output)
        return output