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
    def __init__(self, config):

        super().__init__()
        # load config, config includes all hyperparameters of the run ie dimensions, batch size, number of patches, etc.
        self.config = config

        # 1) Patch projection: map raw 49-D pixels → dim_embed (currently also 49)
        self.patch_proj = nn.Linear(self.config.dim_patch, self.config.dim_in)

        # 2) Fixed 2D sinusoidal positional encoding for a 4×4 grid
        #  Compute once, register as buffer so it’s moved with model but not learned.
        # TODO: un-hard code the grid dims
        self.pos_encoding = nn.Parameter(torch.zeros(1, 16, self.config.dim_in)) #TODO:: add one for CLS token
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # 3) Stack of encoding blocks, now expecting dim_embed in/out
        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(self.config)
            for _ in range(self.config.num_encoders)
        ])

        # 4) Initialise the MLPs
        self.cls_head = MLP(input_dim=self.config.dim_in, hidden_dim=self.config.mlp_hidden_dim, output_dim=10)  # MLP for classification
        self.mlp_between_blocks = MLP(input_dim=self.config.dim_out, hidden_dim=self.config.mlp_hidden_dim, output_dim=self.config.dim_in)  # MLP to apply between encoding blocks        # self.cls_head = nn.Linear(dim_out, 10)  # Classifier head for final output #TODO: add the cls token in
      
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
            nn.GELU(),
            nn.Linear(self.config.dim_out * 4, self.config.dim_out)
        )

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        ### concat all the outputs of the attention heads
        concat = torch.cat(head_outputs, dim=-1)  # Concatenate outputs of all attention heads along the feature dimension

        assert concat.shape[-2:] == (self.config.num_patches, (self.config.num_heads * self.config.dim_out)), f"Expected concatenated output shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.num_heads * self.config.dim_out}), got {concat.shape}"
        ### linear projection of the concatenated output
        out_proj = torch.matmul(concat, self.W_out_proj.weight.t())  # Equivalent to self.W_out_proj(concat) without bias
        assert out_proj.shape[-2:] == (self.config.num_patches, self.config.dim_out), f"Expected output projection shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {out_proj.shape}"

        # 
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

        assert K.shape[-2:] == (self.num_patches, self.dim_proj_QK), f"Expected K shape (16, {self.dim_proj_QK}), got {K.shape}"
        assert V.shape[-2:] == (self.num_patches, self.dim_proj_V), f"Expected V shape (16, {self.dim_proj_V}), got {V.shape}"
        assert Q.shape[-2:] == (self.num_patches, self.dim_proj_QK), f"Expected Q shape (16, {self.dim_proj_QK}), got {Q.shape}"

         ### Scaled dot-product attention
        # Each query row i is dotted with every key row j → a score telling us
        # how much patch i “cares” about patch j.
        # We divide by √d_k to keep the variance of the scores roughly constant
        # regardless of embedding size (otherwise softmax saturates for large d_k).
        
        A = Q @ K.transpose(-2, -1) * (K.size(-1) ** -0.5) # (batch_size, num_patches, num_patches)
        assert A.shape[-2:] == (self.num_patches, self.num_patches), f"Expected A shape ({self.num_patches}, {self.num_patches}), got {A.shape}"
    
      # Softmax turns scores into non-negative weights that sum to 1 along the
        # “from-patch” axis (last dim) – the classic “where should I look?” question.
        ### TODO: check if this is correct, we were trying to avoid nannvalues, maybe this isn't the best place to do this
        attention_scores = torch.clamp(A, min=-30.0, max=30.0)  # Prevent softmax overflow
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Each output patch is now a weighted average of the value vectors, with the
        # weights decided by its query–key similarity.
        attention_out = attention_weights @ V  # (batch_size, num_patches, dim_proj)
        assert attention_out.shape[-2:] == (self.num_patches, self.dim_proj_V), f"Expected attention_out shape ({self.num_patches}, {self.dim_proj_V}), got {attention_out.shape}"
        
        # Linear projection for dimensionality
        output = self.W_h(attention_out)  # (batch_size, num_patches, dim_out)
        assert output.shape[-2:] == (self.num_patches, self.dim_out), f"Expected output shape ({self.num_patches}, {self.dim_out}), got {output.shape}"
        


        # Final linear layer mixes the attended features and sets the channel size
        # expected by the next layer (dim_out).  Nothing fancy, just W_h * x + b.
        return output                     # [B, P, dim_out]
