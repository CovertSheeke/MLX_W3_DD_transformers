import torch
from torch import nn

import math


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ###
        # masked multihead attention
        # self.
        # 

class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        ###
        ## self.masked_multihead_attention = nn.ModuleList(MaskedAttention() for _ in range(num_heads))

        # self.norm1 = nn.LayerNorm(dim_in)

        ### self.multihead_cross_attention = nn.ModuleList(CrossAttention() for _ in range(num_heads))

        # self.norm2 = nn.LayerNorm(dim_in)

        # self.FFN

        # self.norm3 = nn.LayerNorm(dim_in)
        
    def forward(self, x, encoder_output):
        ## masked_head_outputs = [head(x) for head in self.masked_multihead_attention]

        ## concatenation & projection

        ## norm_embeddings = self.norm1(x + projection)

        # cross_head_outputs = [head(norm_embeddings, encoder_output) for head in self.multihead_cross_attention]

        # concatenation & projection

        # norm_embeddings = self.norm2(norm_embeddings + projection)

        # ffn_output = self.FFN(norm_embeddings)

        # norm_embeddings = self.norm3(norm_embeddings + ffn_output)

        # return norm_embeddings  # return the final output after all operations
        pass

class MaskedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the masked attention mechanism here
        self.config = config

        self.dim_in = self.config.dim_in  # Input dimension (e.g., 49 for MNIST patches)
        self.dim_proj_V = self.config.dim_proj_V  # Projection dimension for value matrix (e.g., 49 for MNIST patches)
        self.dim_proj_QK = self.config.dim_proj_QK  # Projection dimension for key and query matrices (e.g., 49 for MNIST patches)
        self.dim_out = self.config.dim_out  # Output dimension (e.g., 49 for MNIST patches)  
        self.num_patches = self.config.num_patches  # Number of patches (e.g., 16 for MNIST)
        self.W_v = torch.nn.Linear(self.dim_in, self.dim_proj_V) ### (49, Y) Y=Z
        self.W_q = torch.nn.Linear(self.dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_k = torch.nn.Linear(self.dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_h = torch.nn.Linear(self.dim_proj_V, self.dim_out) ### (Y, 49)    
        self.dropout = torch.nn.Dropout(p=0.1) 

    def forward(self, x):
        # Implement the forward pass for masked attention
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # calculate attention scores
        d_k: float = math.sqrt(K.size(-1)) 
        attn_scores: torch.Tensor = (Q @ K.transpose(-2, -1)) / d_k  # [B, P, P]
        # mask the attention weights
        mask = torch.triu(torch.ones(attn_weights.size(-2), attn_weights.size(-2)), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))  # upper triangular part
        msk_att = attn_scores + mask.to(attn_scores.device)  # apply mask
        
        attn_weights: torch.Tensor = F.softmax(msk_att, dim=-1)
        
        

        attn_out = msk_att @ V
        
        # softmax/normalise/project
        pass

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the masked attention mechanism here
        self.config = config

        self.dim_in = self.config.dim_in  # Input dimension (e.g., 49 for MNIST patches)
        self.dim_proj_V = self.config.dim_proj_V  # Projection dimension for value matrix (e.g., 49 for MNIST patches)
        self.dim_proj_QK = self.config.dim_proj_QK  # Projection dimension for key and query matrices (e.g., 49 for MNIST patches)
        self.dim_out = self.config.dim_out  # Output dimension (e.g., 49 for MNIST patches)  
        self.num_patches = self.config.num_patches  # Number of patches (e.g., 16 for MNIST)
        self.W_v = torch.nn.Linear(self.dim_in, self.dim_proj_V) ### (49, Y) Y=Z
        self.W_q = torch.nn.Linear(self.dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_k = torch.nn.Linear(self.dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_h = torch.nn.Linear(self.dim_proj_V, self.dim_out) ### (Y, 49)    
        self.dropout = torch.nn.Dropout(p=0.1) 
        
        # Initialize the cross attention mechanism here
        pass

    def forward(self, x, encoder_output):
        # Implement the forward pass for cross attention
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)
        attn_scores: torch.Tensor = (Q @ K.transpose(-2, -1))
        attn_scores *= (K.size(-1) ** -0.5) # [B, P, P]
        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)
        
        
        
        H = attn_weights @ V
        # normalise/project

        pass