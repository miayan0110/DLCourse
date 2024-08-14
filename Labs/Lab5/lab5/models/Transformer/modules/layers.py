import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop)

        self.w_o = nn.Linear(dim, dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        # reference: 
        # https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
        # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
        batch_size = x.shape[0]

        query = self.w_q(x)
        key = self.w_q(x)
        value = self.w_q(x)

        query = query.view(batch_size, -1, self.num_heads, self.dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        dk = key.size()[-1]
        attention_weights = self.softmax(matmul_qk / (dk ** 0.5))
        dropout_attention_weights = self.dropout(attention_weights)
        output = torch.matmul(dropout_attention_weights, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.w_o(output)
        return output
        # raise Exception('TODO1!')

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    