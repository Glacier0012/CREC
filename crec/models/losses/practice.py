import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input, num_heads, dim_output, dropout_prob=0.5):
        super(MultiHeadSelfAttention, self).__init__
        hidden_size = input.size(-1)
        self.num_heads = num_heads
        self.heads_size = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = F.softmax()
    
    def forward(self, x, mask):
        B = x.size(0)
        L = x.size(1)

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.reshape(B, L, self.num_heads, self.heads_size).transpose(1,2)
        k = k.reshape(B, L, self.num_heads, self.heads_size).transpose(1,2)
        v = v.reshape(B, L, self.num_heads, self.heads_size).transpose(1,2)

        attn_score = q @ k.transpose(-2,-1) / (self.heads_size ** 0.5)
        if mask is not None:
            attn_score = attn_score.mask_fill(mask == 0, 1e-9)
        attn_score = nn.Softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)

        output = attn_score @ v
        output = output.transpose(1,2).reshape(B, L, -1)

        output = self.W_o(output)
        return output


        