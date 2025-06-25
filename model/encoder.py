import torch
import torch.nn.functional as F
import torch.nn as nn
import math 
import numpy as np
from patch_and_embed import image_to_patch_columns

# Function to get 2D sinusoidal positional encodings
#TODO: NO longer used, we used learned positional encodings instead. Could remove.
def get_2d_sincos_pos_enc(grid_h: int, grid_w: int, d_model: int) -> torch.Tensor:
    """
    Return a [grid_h*grid_w, d_model] tensor of fixed 2D sinusoidal positional encodings.
    Splits into row/column parts; handles odd d_model by uneven split.
    So the first hald of the embedding gets the row position encoding added to the tensor, the 
    second half gets the column position encoding added.
    """
    # how many dims for row vs col
    half1 = d_model // 2                 # dims for the row signal
    half2 = d_model - half1              # dims for the col signal

    # Frequencies for rows and cols
    div_r = 10000 ** (torch.arange(half1, dtype=torch.float32) / half1)
    div_c = 10000 ** (torch.arange(half2, dtype=torch.float32) / half2)

    # Row/col positions
    pos_r = torch.arange(grid_h, dtype=torch.float32)[:, None]  # [H,1]
    pos_c = torch.arange(grid_w, dtype=torch.float32)[:, None]  # [W,1]

    # [H, half1] and [W, half2]
    pe_r = torch.cat(
        (torch.sin(pos_r / div_r[::2]), torch.cos(pos_r / div_r[1::2])), dim=1
    )
    pe_c = torch.cat(
        (torch.sin(pos_c / div_c[::2]), torch.cos(pos_c / div_c[1::2])), dim=1
    )

    # Broadcast & interleave row/col encodings → [H, W, d_model]
    pe = torch.zeros(grid_h, grid_w, d_model)
    pe[:, :, :half1] = pe_r[:, None, :]
    pe[:, :, half1:] = pe_c[None, :, :]

    return pe.view(grid_h * grid_w, d_model)


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
    def __init__(self, 
                 config,
                 dim_in=49, 
                 dim_embed=49, #after projecting the input to the embedding dimension
                 dim_proj=49, 
                 dim_out=49, 
                 num_heads=8):
        
        super().__init__()
        # load config, config includes all hyperparameters of the run ie dimensions, batch size, number of patches, etc.
        self.config = config

        # 1) Patch projection: map raw 49-D pixels → dim_embed (currently also 49)
        self.patch_proj = nn.Linear(dim_in, dim_embed)

        # 2) Add pos embedding
        # option a: Fixed 2D sinusoidal positional encoding for a 4×4 grid
        #pos_enc = get_2d_sincos_pos_enc(grid_h=4, grid_w=4, d_model=dim_embed)  # [16, dim_embed]
        # option b: learnable positional encoding, randomly initialised
        self.pos_encoding = nn.Parameter(torch.zeros(1, 16, dim_embed)) #TODO:: add one for CLS token
        torch.nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # 3) Stack of encoding blocks, now expecting dim_embed in/out
        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(self.config,
                          dim_in=dim_embed,
                          dim_proj=dim_embed,
                          dim_out=dim_out,
                          num_heads=self.config.num_heads)
            for _ in range(self.config.num_encoders)
        ])

        # 4) Initialise the MLPs
        self.cls_head = MLP(input_dim=49, hidden_dim=25, output_dim=10)  # MLP for classification
        self.mlp_between_blocks = MLP(input_dim=49, hidden_dim=49, output_dim=49)  # MLP to apply between encoding blocks        # self.cls_head = nn.Linear(dim_out, 10)  # Classifier head for final output #TODO: add the cls token in
      
    def forward(self, x, trg):
        """
        Args:
        embedding: [batch, num_patches, dim_in] raw patch pixels.
        target_labels: [batch] tensor of class indices (0-9).
        Returns:
        Cross-entropy loss.
        """
        # project raw patches into embed-space, then add learned pos-enc
        x = self.patch_proj(x)                          # (B, P, dim_embed)
        x = x + self.pos_encoding[:, :x.size(1), :]     # (B, P, dim_embed)
        x_n = x

        for encoding_block in self.encoding_blocks:
            x_n = encoding_block(x_n) # B, num_patches, dim_proj_V
            assert x_n.shape[-2:] == (self.config.num_patches, self.config.dim_proj_V), f"Expected x_n shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_proj_V}), got {x_n.shape}"
            x_n = self.mlp_between_blocks(x_n) # B, num_patches, dim_out
            assert x_n.shape[-2:] == (self.config.num_patches, self.config.dim_out), f"Expected x_n shape ({self.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {x_n.shape}"

        pooled = x_n.mean(dim=1) # Average pooling over the num_patches dimension: B, dim_out
        assert pooled.shape[-1:] == torch.Size([self.config.dim_out]), f"Expected pooled shape ({self.config.batch_size}, {self.config.dim_out}), got {pooled.shape}"
        predictions = self.cls_head(pooled)  # Assuming self.mlp is defined in the class
        assert predictions.shape[-1:] == torch.Size([10]), f"Expected predictions shape ({self.config.batch_size}, 10), got {predictions.shape}"
        
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == trg).float().sum()
        accuracy = correct / predictions.shape[0]
        loss = F.cross_entropy(predictions, trg)
        # Compute cross-entropy loss
        return loss, accuracy
        


        # target_labels should be class indices (LongTensor), not one-hot encoded
        # return F.cross_entropy(logits, target_labels)

class EncodingBlock(torch.nn.Module):
    def __init__(self, config, dim_in=49, dim_proj=49, dim_out=49, num_heads=8):
        super().__init__()
        self.config = config
        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(self.config, dim_in=dim_in, dim_proj=dim_in, dim_out=dim_out) for _ in range(num_heads)
        ])
        # TODO add the project, layer norm and feedforward to diagram
        # Linear projection after concatenation of attention heads
        self.W_out_proj = nn.Linear(num_heads * dim_out, dim_out)  
        # add layer normalization
        self.layernorm1 = nn.LayerNorm(dim_out)
        self.layernorm2 = nn.LayerNorm(dim_out)
        # feedforward: typically expands then back
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4),
            nn.GELU(),
            nn.Linear(dim_out * 4, dim_out)
        )

    def forward(self, embedding):
        # image_columns = image_to_patch_columns  # Assuming image is already embedded
        head_outputs = [head(embedding) for head in self.heads]

        ### concat all the outputs of the attention heads
        concat = torch.cat(head_outputs, dim=-1)  # Concatenate outputs of all attention heads along the feature dimension
        # D linear project of the concatenated outputs to a new dim 
        attn_out = self.W_out_proj(concat) 
        # 
        projected_embed = self.layernorm1(embedding + attn_out)
        # feed forward
        ffn_out = self.ffn(projected_embed)
        # Add layer normalization after the feedforward network
        out = self.layernorm2(projected_embed + ffn_out)
        return out

class SelfAttentionHead(torch.nn.Module):
    def __init__(self, config, dim_in, dim_proj, dim_out): ### TODO: seperate dim_v, dim_qk
        ### TODO: include batch size in assertions
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_proj = dim_proj
        self.dim_out = dim_out  
        self.W_q = nn.Linear(dim_in, dim_proj)
        self.W_k = nn.Linear(dim_in, dim_proj)
        self.W_v = nn.Linear(dim_in, dim_proj)
        self.W_h = nn.Linear(dim_proj, dim_out)   
    
    def forward(self, embedding):
        """
        Forward pass of the self-attention head.
        Args:
            embedding (torch.Tensor): Input tensor of shape (batch_size, num_patches, dim_in)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, dim_out)
        """
        # Compute queries, keys, and values
        Q = self.W_q(embedding)
        K = self.W_k(embedding)
        V = self.W_v(embedding)
        ## Debugging shapes
        # print(f"Embedding shape: {embedding.shape}")
        # print(f"Q shape: {Q.shape}")
        # print(f"K shape: {K.shape}")
        # print(f"V shape: {V.shape}")
        # assert K.shape == (16, self.dim_proj), f"Expected K shape (16, {self.dim_proj}), got {K.shape}"
        # assert V.shape == (16, self.dim_proj), f"Expected V shape (16, {self.dim_proj}), got {V.shape}"
        # assert Q.shape == (16, self.dim_proj), f"Expected Q shape (16, {self.dim_proj}), got {Q.shape}"

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

        # Each output patch is now a weighted average of the value vectors, with the
        # weights decided by its query–key similarity.
        attn_out: torch.Tensor = attn_weights @ V     # [B, P, dim_proj]
        
        # Debugging output shape
        # print(f"Output shape: {attn_out.shape}")
        # assert attn_out.shape == (16, 49), f"Expected output shape (16, 49), got {attn_out.shape}"

        # Final linear layer mixes the attended features and sets the channel size
        # expected by the next layer (dim_out).  Nothing fancy, just W_h * x + b.
        return self.W_h(attn_out)                     # [B, P, dim_out]
