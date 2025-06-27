import torch
from torch import nn
import torch.nn.functional as F


import math


TOKEN2IDX = {
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
  "6": 6,
  "7": 7,
  "8": 8,
  "9": 9,
  "<pad>": 10,
  "<start>": 11,
  "<stop>": 12,
}
IDX2TOKEN = {v: k for k, v in TOKEN2IDX.items()}



class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Initialise embeddings for each token in the vocab
        self.token_emb = nn.Embedding(len(IDX2TOKEN),
                                  config.dec_dim_in)  # learnable
        
        # Initialise positional embeddings for each position in the sequence
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dec_dim_in)

        # Initialise the linear layer to project the input to the output vocab size
        self.to_logits = nn.Linear(config.dec_dim_in, len(IDX2TOKEN))

        # Initialise the decoder blocks
        self.decoding_blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.num_decoders)
        ])

    def forward(self, tokens, enc_out):
        """
        Args:
            tokens: (B, T) integer IDs of input tokens for the decoder.
            - B = batch size (number of images)
            - T = max seq len, so max num of tokens in a sequence.
            - Each entry is an index, e.g: <s>, 3, 7, </s>, <pad>, <pad>, ...
            
            enc_out: (B, P+1, dim_in) output from the encoder.
            - P = number of image patches
            - +1 accounts for the CLS token

        Returns:
            logits: (B, T, vocab_size)
            - predicted logits for the next token at each position
        """
        # B,
        T = tokens.shape  # B: batch size, T: token sequence length
        # 1. Embed tokens and add positional embeddings
        # print('tokens', tokens.shape)  # (B, T)
        position_ids = torch.arange(self.config.max_seq_len, device=tokens.device)

        pos = self.pos_emb(position_ids).unsqueeze(0)   # → (1, T, D)
        x_n = self.token_emb(tokens) + pos              # broadcasts to (B, T, D)

        # 2. Pass through the decoder blocks
        for decoding_block in self.decoding_blocks:
            x_n = decoding_block(x_n, enc_out)  
 
        # Project final output from decoder blocks to logits
        return self.to_logits(x_n)  # (B, T, vocab_size)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.mask_heads = nn.ModuleList([
            MaskedAttention(config) for _ in range(config.dec_mask_num_heads)
        ])

        self.mask_W_out_proj = torch.nn.Linear(config.dec_mask_num_heads*config.dec_dim_out, config.dec_dim_in)  # projection to combine outputs of masked heads
        self.norm1 = nn.LayerNorm(config.dec_dim_in)

        self.cross_heads = nn.ModuleList([
            CrossAttention(config) for _ in range(config.dec_cross_num_heads)    
        ])

        self.cross_W_out_proj = torch.nn.Linear(config.dec_cross_num_heads*config.dec_dim_out, config.dec_dim_in)  
        self.norm2 = nn.LayerNorm(config.dec_dim_in)

        self.ffn = nn.Sequential(
            nn.Linear(config.dec_dim_in, config.dec_dim_in*4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(config.dec_dim_in*4, config.dec_dim_in)
            # could be dec_dim_out, but might also work
        )

        self.norm3 = nn.LayerNorm(config.dec_dim_in)
        
    def forward(self, x, encoder_output):
        mask_head_outputs = [head(x) for head in self.mask_heads]

        ## concatenation & projection
        concat = torch.cat(mask_head_outputs, dim=-1)
        out_proj = self.mask_W_out_proj(concat)

        norm_emb = self.norm1(x + out_proj)

        cross_head_outputs = [head(norm_emb, encoder_output) for head in self.cross_heads]

        # concatenation & projection
        concat = torch.cat(cross_head_outputs, dim=-1)
        out_proj = self.cross_W_out_proj(concat)

        # norm_embeddings = self.norm2(norm_embeddings + projection)

        norm_emb = self.norm2(norm_emb + out_proj)

        ffn_output = self.ffn(norm_emb)

        out = self.norm3(norm_emb + ffn_output)

        # return norm_embeddings  # return the final output after all operations
        return out

class MaskedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the masked attention mechanism here
        self.config = config

        self.dec_dim_in = self.config.dec_dim_in  # Input dimension (e.g., 49 for MNIST patches)
        self.dim_proj_V = self.config.dim_proj_V  # Projection dimension for value matrix (e.g., 49 for MNIST patches)
        self.dim_proj_QK = self.config.dim_proj_QK  # Projection dimension for key and query matrices (e.g., 49 for MNIST patches)
        self.dec_dim_out = self.config.dec_dim_out  # Output dimension (e.g., 49 for MNIST patches)  
        self.W_v = torch.nn.Linear(self.dec_dim_in, self.dim_proj_V) ### (49, Y) Y=Z
        self.W_q = torch.nn.Linear(self.dec_dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_k = torch.nn.Linear(self.dec_dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_h = torch.nn.Linear(self.dim_proj_V, self.dec_dim_out) ### (Y, 49)    
        self.dropout = torch.nn.Dropout(p=0.1) 

    def forward(self, x):
        # Implement the forward pass for masked attention
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # calculate attention scores
        d_k: float = math.sqrt(K.size(-1)) 
        attn_scores: torch.Tensor = (Q @ K.transpose(-2, -1)) / d_k  # [B, P, P]
        
        # build a clean causal mask of shape (T, T) on the right device
        B, T, _ = x.shape
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # apply it in one go: any [i,j] where j>i becomes -inf
        masked_scores = attn_scores.masked_fill(causal.unsqueeze(0), float("-inf"))
        
        # normalise
        attn_weights = F.softmax(masked_scores, dim=-1)

        # apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)  # [B, P, P

        attn_out = attn_weights @ V
        
        # softmax/normalise/project. TODO: don't think we need for now but see later.
        return self.W_h(attn_out)  # [B, P, dim_out] - project to output dimension

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the masked attention mechanism here
        self.config = config

        self.dec_dim_in = self.config.dec_dim_in  # Input dimension (e.g., 49 for MNIST patches)
        self.dim_proj_V = self.config.dim_proj_V  # Projection dimension for value matrix (e.g., 49 for MNIST patches)
        self.dim_proj_QK = self.config.dim_proj_QK  # Projection dimension for key and query matrices (e.g., 49 for MNIST patches)
        self.dec_dim_out = self.config.dec_dim_out  # Output dimension (e.g., 49 for MNIST patches)  
        self.W_v = torch.nn.Linear(self.dec_dim_in, self.dim_proj_V) ### (49, Y) Y=Z
        self.W_q = torch.nn.Linear(self.dec_dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_k = torch.nn.Linear(self.dec_dim_in, self.dim_proj_QK) ### (49, Z)
        self.W_h = torch.nn.Linear(self.dim_proj_V, self.dec_dim_out) ### (Y, 49)    
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, encoder_output):
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)
        attn_scores: torch.Tensor = (Q @ K.transpose(-2, -1))
        attn_scores *= (K.size(-1) ** -0.5) # [B, P, P]
        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)
        # apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)  # [B, P, P

        H = attn_weights @ V
        
        # softmax/normalise/project. TODO: don't think we need for now but see later.
        return self.W_h(H)  # [B, P, dim_out] - project to output dimension