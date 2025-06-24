import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from patch_and_embed import image_to_patch_columns

# Function to get 2D sinusoidal positional encodings
def get_2d_sincos_pos_enc(grid_h: int, grid_w: int, d_model: int) -> torch.Tensor:
    """
    Return a [grid_h*grid_w, d_model] tensor of fixed 2D sinusoidal positional encodings.
    Splits into row/column parts; handles odd d_model by uneven split.
    So the first hald of the embedding gets the row position encoding added to the tensor, the 
    second half gets the column position encoding added.
    """
    # how many dims for row vs col
    half1 = d_model // 2
    half2 = d_model - half1
    # how many even/odd slots in each half
    even1 = (half1 + 1) // 2
    odd1  = half1 // 2
    even2 = (half2 + 1) // 2
    odd2  = half2 // 2

    # row and col indices
    pos_r = torch.arange(grid_h).unsqueeze(1).float()  # [H,1]
    pos_c = torch.arange(grid_w).unsqueeze(1).float()  # [W,1]

    # frequency terms
    div_r_even = torch.exp(torch.arange(even1).float() * (-np.log(10000.0) / half1))
    div_r_odd  = torch.exp(torch.arange(odd1).float()  * (-np.log(10000.0) / half1))
    div_c_even = torch.exp(torch.arange(even2).float() * (-np.log(10000.0) / half2))
    div_c_odd  = torch.exp(torch.arange(odd2).float()  * (-np.log(10000.0) / half2))

    # build row & col PEs
    row_pe = torch.zeros(grid_h, half1)
    row_pe[:, 0::2] = torch.sin(pos_r * div_r_even)
    row_pe[:, 1::2] = torch.cos(pos_r * div_r_odd)

    col_pe = torch.zeros(grid_w, half2)
    col_pe[:, 0::2] = torch.sin(pos_c * div_c_even)
    col_pe[:, 1::2] = torch.cos(pos_c * div_c_odd)

    # combine into [H, W, d_model]
    pe = torch.zeros(grid_h, grid_w, d_model)
    for i in range(grid_h):
        for j in range(grid_w):
            pe[i, j, :half1]  = row_pe[i]
            pe[i, j, half1:] = col_pe[j]

    return pe.view(grid_h * grid_w, d_model)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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

        # 2) Fixed 2D sinusoidal positional encoding for a 4×4 grid
        #  Compute once, register as buffer so it’s moved with model but not learned.
        # TODO: un-hard code the grid dims
        pos_enc = get_2d_sincos_pos_enc(grid_h=4, grid_w=4, d_model=dim_embed)  # [16, dim_embed]
        self.register_buffer("pos_encoding", pos_enc.unsqueeze(0))  # shape [1,16,dim_embed]

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
      
    def forward(self, embedding, target_labels):
        """
        Args:
        embedding: [batch, num_patches, dim_in] raw patch pixels.
        target_labels: [batch] tensor of class indices (0-9).
        Returns:
        Cross-entropy loss.
        """
        embedding_n = embedding
        # Project raw pixels to a higher-dimensional embedding space
        # (although for now we still use 49) #TODO: select a better dim (e.g 64)
        embedding_n = self.patch_proj(embedding)

        # Add positional encoding to the embedding
        embedding_n = embedding_n + self.pos_encoding 

        for encoding_block in self.encoding_blocks:
            embedding_n = encoding_block(embedding_n) # B, num_patches, dim_proj_V
            assert embedding_n.shape[-2:] == (self.config.num_patches, self.config.dim_proj_V), f"Expected embedding_n shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_proj_V}), got {embedding_n.shape}"
            embedding_n = self.mlp_between_blocks(embedding_n) # B, num_patches, dim_out
            assert embedding_n.shape[-2:] == (self.config.num_patches, self.config.dim_out), f"Expected embedding_n shape ({self.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {embedding_n.shape}"

        pooled = embedding_n.mean(dim=1) # Average pooling over the num_patches dimension: B, dim_out
        #assert pooled.shape == torch.Size([, self.config.dim_out]), f"Expected pooled shape ({self.config.batch_size}, {self.config.dim_out}), got {pooled.shape}"
        logits = self.cls_head(pooled)  # Assuming self.mlp is defined in the class
        #assert logits.shape == torch.Size([self.config.batch_size, 10]), f"Expected logits shape ({self.config.batch_size}, 10), got {logits.shape}"
        ### TODO: apply normalisation to prediction (softmax?)

        # logits = self.cls_head(embedding_n)  # Classifier head for final output
        # print(f"logits shape: {logits.shape}")
        # Average pool over the 16 (num_patches) dimension to get shape (batch_size, dim_out)
        # print(f"Pooled logits shape: {pooled.shape}")

        # Compute cross-entropy loss
        return F.cross_entropy(logits, target_labels)
        


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


        # Q = torch.matmul(embedding, self.W_q)
        # K = torch.matmul(embedding, self.W_k)
        # V = torch.matmul(embedding, self.W_v)

        # Compute attention scores
        A = Q @ K.transpose(-2, -1)  # (batch_size, num_patches, num_patches)
        # debug shape
        # print(f"A shape: {A.shape}")
        # assert A.shape == (16, 16), f"Expected A shape (16, 16), got {A.shape}"
        ### TODO: maybe remove the epsilon
        eps = 1e-6  # Small epsilon to avoid division by zero
        scale = np.sqrt(K.size(-1)) + eps
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        ### TODO: check if this is correct, we were trying to avoid nannvalues, maybe this isn't the best place to do this
        attention_scores = torch.clamp(attention_scores, min=-30.0, max=30.0)  # Prevent softmax overflow
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute output
        attention_out = attention_weights @ V  # (batch_size, num_patches, dim_proj)
        
        # Linear projection for dimensionality
        output = self.W_h(attention_out)  # (batch_size, num_patches, dim_out)
        # Debugging output shape
        # print(f"Output shape: {output.shape}")
        # assert output.shape == (16, 49), f"Expected output shape (16, 49), got {output.shape}"

        return output
