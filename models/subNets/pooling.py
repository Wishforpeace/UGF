import torch.nn as nn

import torch


class MultiheadSelfAttentionWithPooling(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadSelfAttentionWithPooling, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.embed_size = embed_size

    def forward(self, x):
        # Transpose x to match the input shape requirement for nn.MultiheadAttention: (sequence_length, batch_size, embed_size)
        x = x.transpose(0, 1)
        attention_output, _ = self.multihead_attention(x, x, x)
        # Transpose back to (batch_size, sequence_length, embed_size)
        attention_output = attention_output.transpose(0, 1)
        # Average pooling across the sequence dimension
        pooled_output,_ = torch.max(attention_output, dim=1)
        return pooled_output
    