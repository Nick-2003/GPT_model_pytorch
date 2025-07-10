import torch

from model_building_blocks import (TokenAndPositionEmbedding, TransformerBlock,
                                   create_attention_mask)


class GPTModel(torch.nn.Module):
    """GPT Model Class with Multiple Transformer Layers."""

    def __init__(
        self,
        max_len_input: int,
        vocab_size: int,
        embed_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        key_dim: int,
        num_layers: int = 1,  # New parameter for number of transformer layers
    ) -> None:
        """Init Function."""
        super().__init__()
        
        self.embedding_layer = TokenAndPositionEmbedding(
            max_len_input=max_len_input,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )

        # self.transformer = TransformerBlock(
        #     num_heads=num_heads,
        #     key_dim=key_dim,
        #     embed_dim=embed_dim,
        #     ff_dim=feed_forward_dim,
        #     mask_function=create_attention_mask,
        # )
        
        # Method 1: Using ModuleList (recommended for flexibility)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                embed_dim=embed_dim,
                ff_dim=feed_forward_dim,
                mask_function=create_attention_mask,
            ) for _ in range(num_layers)
        ])
        
        # Alternative Method 2: Using Sequential (simpler but less flexible)
        # self.transformer_blocks = torch.nn.Sequential(*[
        #     TransformerBlock(
        #         num_heads=num_heads,
        #         key_dim=key_dim,
        #         embed_dim=embed_dim,
        #         ff_dim=feed_forward_dim,
        #         mask_function=create_attention_mask,
        #     ) for _ in range(num_layers)
        # ])
        
        self.output_layer = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        x = self.embedding_layer(input_tensor)
        
        # Method 1: Forward through ModuleList
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Alternative Method 2: Forward through Sequential
        # x = self.transformer_blocks(x)
        
        output = self.output_layer(x)
        return output


# # Alternative implementation with additional features
# class GPTModelAdvanced(torch.nn.Module):
#     """Advanced GPT Model with configurable layer parameters."""

#     def __init__(
#         self,
#         max_len_input: int,
#         vocab_size: int,
#         embed_dim: int,
#         feed_forward_dim: int,
#         num_heads: int,
#         key_dim: int,
#         num_layers: int = 6,
#         dropout_rate: float = 0.1,
#         layer_norm_eps: float = 1e-6,
#     ) -> None:
#         """Init Function with advanced options."""
#         super().__init__()
        
#         self.embedding_layer = TokenAndPositionEmbedding(
#             max_len_input=max_len_input,
#             vocab_size=vocab_size,
#             embed_dim=embed_dim,
#         )
        
#         # Optional: Add dropout after embedding
#         self.embedding_dropout = torch.nn.Dropout(dropout_rate)
        
#         # Create transformer blocks
#         self.transformer_blocks = torch.nn.ModuleList([
#             TransformerBlock(
#                 num_heads=num_heads,
#                 key_dim=key_dim,
#                 embed_dim=embed_dim,
#                 ff_dim=feed_forward_dim,
#                 mask_function=create_attention_mask,
#                 # Add any additional parameters your TransformerBlock supports
#             ) for _ in range(num_layers)
#         ])
        
#         # Optional: Add final layer normalization
#         self.final_layer_norm = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
#         self.output_layer = torch.nn.Linear(embed_dim, vocab_size)

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         """Forward Pass."""
#         x = self.embedding_layer(input_tensor)
#         x = self.embedding_dropout(x)
        
#         # Forward through all transformer blocks
#         for transformer_block in self.transformer_blocks:
#             x = transformer_block(x)
        
#         # Optional: Apply final layer normalization
#         x = self.final_layer_norm(x)
        
#         output = self.output_layer(x)
#         return output


# # Usage example
# if __name__ == "__main__":
#     # Example configuration
#     model_config = {
#         "max_len_input": 512,
#         "vocab_size": 50257,
#         "embed_dim": 768,
#         "feed_forward_dim": 3072,
#         "num_heads": 12,
#         "key_dim": 64,
#         "num_layers": 12,  # 12 transformer layers
#     }
    
#     # Create model
#     model = GPTModel(**model_config)
    
#     # Print model info
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     # Test forward pass
#     batch_size = 2
#     seq_len = 256
#     dummy_input = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len))
    
#     with torch.no_grad():
#         output = model(dummy_input)
#         print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, vocab_size)