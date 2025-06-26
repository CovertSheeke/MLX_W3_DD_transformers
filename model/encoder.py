import torch
import torch.nn.functional as F
import torch.nn as nn
import math 
import numpy as np
from patch_and_embed import image_to_patch_columns

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):

        super().__init__()
        # load config, config includes all hyperparameters of the run ie dimensions, batch size, number of patches, etc.
        self.config = config

        # 1) patch → embed
        self.patch_proj = nn.Linear(config.dim_patch, config.dim_in)

        # 2) learnable CLS token + pos-encoding for P patches + 1 CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim_in))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Create a learnable positional encoding and add 1 dim for the cls token
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.dim_in)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # 3) transformer blocks
        self.encoding_blocks = nn.ModuleList([
            EncodingBlock(config) for _ in range(config.num_encoders)
        ])

        # 4) heads
        # MLP between blocks still uses dim_out→dim_in
        self.mlp_between_blocks = MLP(
            input_dim=config.dim_out,
            hidden_dim=config.mlp_hidden_dim,
            output_dim=config.dim_in,
        )
        # final CLS-head
        self.cls_head = MLP(
            input_dim=config.dim_out,
            hidden_dim=config.mlp_hidden_dim,
            output_dim=10,
        )

        # 5) dropout and normalization for final output
        self.final_norm = nn.LayerNorm(config.dim_out)
        self.final_dropout = nn.Dropout(p=0.1)

    def forward(self, x_n):
        """
        Args:
        embedding: [batch, num_patches, dim_in] raw patch pixels.
        target_labels: [batch] tensor of class indices (0-9).
        Returns:
        Cross-entropy loss.
        """
        # project raw patches into embed-space, then add learned pos-enc
        bsz = x_n.size(0)
        # project patches
        x_n = self.patch_proj(x_n)                                # → (B, P, dim_in)
        # prepended CLS
        cls = self.cls_token.expand(bsz, -1, -1)              # → (B, 1, dim_in)
        x_n = torch.cat((cls, x_n), dim=1)                        # → (B, P+1, dim_in)
        # add pos encoding
        x_n = x_n + self.pos_encoding[:, : x_n.size(1), :]          # still (B, P+1, dim_in)

        # Loop through encoding blocks
        for encoding_block in self.encoding_blocks:
            x_n = encoding_block(x_n) # B, num_patches, dim_proj_V
            #assert x_n.shape[-2:] == (self.config.num_patches +1, self.config.dim_proj_V), f"Expected x_n shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_proj_V}), got {x_n.shape}"
            x_n = self.mlp_between_blocks(x_n) # B, num_patches, dim_out
            #assert x_n.shape[-2:] == (self.config.num_patches +1, self.config.dim_out), f"Expected x_n shape ({self.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {x_n.shape}"

        return x_n

class EncodingBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(self.config) for _ in range(self.config.num_heads)
        ])

        # TODO add the project, layer norm and feedforward to diagram
        # Linear projection after concatenation of attention heads
        self.W_out_proj = torch.nn.Linear(self.config.num_heads*self.config.dim_out, self.config.dim_out)  
        # add layer normalization
        self.layernorm1 = nn.LayerNorm(self.config.dim_out)  # Normalization after attention
        self.layernorm2 = nn.LayerNorm(self.config.dim_out)  # Normalization after feedforward
        # feedforward: typically expands then back
        self.ffn = nn.Sequential(
            nn.Linear(self.config.dim_out, self.config.dim_out * 4),
            nn.ReLU(),
            nn.Dropout(p=0.1), 
            nn.Linear(self.config.dim_out * 4, self.config.dim_out)
        )

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        ### concat all the outputs of the attention heads
        concat = torch.cat(head_outputs, dim=-1)  # Concatenate outputs of all attention heads along the feature dimension

        assert concat.shape[-2:] == (self.config.num_patches +1, (self.config.num_heads * self.config.dim_out)), f"Expected concatenated output shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.num_heads * self.config.dim_out}), got {concat.shape}"
        ### linear projection of the concatenated output
        out_proj = self.W_out_proj(concat)
        assert out_proj.shape[-2:] == (self.config.num_patches +1, self.config.dim_out), f"Expected output projection shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {out_proj.shape}"

        # Add residual connection and layer normalization after the attention heads
        norm_emb = self.layernorm1(x + out_proj)
        # feed forward
        ffn_out = self.ffn(norm_emb)
        # Add layer normalization after the feedforward network
        out = self.layernorm2(norm_emb + ffn_out)
        return out

class SelfAttentionHead(torch.nn.Module):
    def __init__(self, config): ### TODO: seperate dim_v, dim_qk
        ### TODO: include batch size in assertions
        super().__init__()
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
        self.dropout = torch.nn.Dropout(p=0.1)  # Dropout layer for regularization

    
    def forward(self, x):
        """
        Forward pass of the self-attention head.
        Args:
            embedding (torch.Tensor): Input tensor of shape (batch_size, num_patches, dim_in)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, dim_out)
        """
        # Compute queries, keys, and values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        ## Debugging shapes
        # assert K.shape[-2:] == (self.num_patches, self.dim_proj_QK), f"Expected K shape (16, {self.dim_proj_QK}), got {K.shape}"
        # assert V.shape[-2:] == (self.num_patches, self.dim_proj_V), f"Expected V shape (16, {self.dim_proj_V}), got {V.shape}"
        # assert Q.shape[-2:] == (self.num_patches, self.dim_proj_QK), f"Expected Q shape (16, {self.dim_proj_QK}), got {Q.shape}"

        ### Scaled dot-product attention
        # Each query row i is dotted with every key row j → a score telling us
        # how much patch i “cares” about patch j.
        # We divide by √d_k to keep the variance of the scores roughly constant
        # regardless of embedding size (otherwise softmax saturates for large d_k).
        d_k: float = math.sqrt(K.size(-1))            # dim_proj as a scalar
        attn_scores: torch.Tensor = (Q @ K.transpose(-2, -1)) / d_k  # [B, P, P]
        # Softmax turns scores into non-negative weights that sum to 1 along the
        # “from-patch” axis (last dim) – the classic “where should I look?” question.
        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)  # [B, P, P]

        # Add the dropout
        attn_weights = self.dropout(attn_weights)  # [B, P, P]
        # Each output patch is now a weighted average of the value vectors, with the
        # weights decided by its query–key similarity.
        attn_out: torch.Tensor = attn_weights @ V     # [B, P, dim_proj]

        # Debugging output shape
        # print(f"Output shape: {attn_out.shape}")
        # assert attn_out.shape == (16, 49), f"Expected output shape (16, 49), got {attn_out.shape}"

        # Final linear layer mixes the attended features and sets the channel size
        # expected by the next layer (dim_out).  Nothing fancy, just W_h * x + b.
        return self.W_h(attn_out)                     # [B, P, dim_out]