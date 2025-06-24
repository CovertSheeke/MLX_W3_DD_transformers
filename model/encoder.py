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
    # split sizes
    half1 = d_model // 2
    half2 = d_model - half1
    # prepare row and column indices
    position_row = torch.arange(grid_h).unsqueeze(1).float()  # shape [H,1]
    position_col = torch.arange(grid_w).unsqueeze(1).float()  # shape [W,1]
    # compute div terms
    # for row part: size half1
    div_term_row = torch.exp(torch.arange(0, half1, 2).float() * (-np.log(10000.0) / half1))
    # for col part: size half2
    div_term_col = torch.exp(torch.arange(0, half2, 2).float() * (-np.log(10000.0) / half2))
    # allocate
    row_positional_enc = torch.zeros(grid_h, half1)
    col_positional_enc = torch.zeros(grid_w, half2)
    # fill row position encoding using sine and cosine
    row_positional_enc[:, 0::2] = torch.sin(position_row * div_term_row)
    row_positional_enc[:, 1::2] = torch.cos(position_row * div_term_row)
    # fill col position encoding
    col_positional_enc[:, 0::2] = torch.sin(position_col * div_term_col)
    col_positional_enc[:, 1::2] = torch.cos(position_col * div_term_col)
    # combine into [H, W, d_model]
    positional_enc = torch.zeros(grid_h, grid_w, d_model)
    for i in range(grid_h):
        for j in range(grid_w):
            positional_enc[i, j, :half1] = row_positional_enc[i]
            positional_enc[i, j, half1:] = col_positional_enc[j]
    # flatten to [H*W, d_model]
    return positional_enc.view(grid_h * grid_w, d_model)


class MLP(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=25, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, 
                 dim_in=49, 
                 dim_embed=49, #after projecting the input to the embedding dimension
                 dim_proj=49, 
                 dim_out=49, 
                 num_heads=8, 
                 num_encoders=6):
        
        super().__init__()

        # 1) Patch projection: map raw 49-D pixels → dim_embed (currently also 49)
        self.patch_proj = nn.Linear(dim_in, dim_embed)

        # 2) Fixed 2D sinusoidal positional encoding for a 4×4 grid
        #  Compute once, register as buffer so it’s moved with model but not learned.
        # TODO: un hard code these.
        pe = get_2d_sincos_pos_enc(grid_h=4, grid_w=4, d_model=dim_embed)  # [16, dim_embed]
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # shape [1,16,dim_embed]

        # 3) Stack of encoding blocks, now expecting dim_embed in/out
        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(dim_in=dim_embed,
                          dim_proj=dim_embed,
                          dim_out=dim_out,
                          num_heads=num_heads)
            for _ in range(num_encoders)
        ])

        # 4) Final MLP for classification: average pooled embedding → logits
        self.mlp = MLP(input_dim=dim_out, hidden_dim=25, output_dim=10)
        # self.cls_head = nn.Linear(dim_out, 10)  # Classifier head for final output #TODO: add the cls token in
      
    def forward(self, embedding, target_labels):
        embedding_n = embedding
        for encoding_block in self.encoding_blocks:
            embedding_n = encoding_block(embedding_n)
            ### TODO: add MLP to the output of each block?

        # print(f"Final embedding_n shape before loss: {embedding_n.shape}")
        # print(f"Target labels shape: {target_labels.shape}")

        pooled = embedding_n.mean(dim=1)
        # loss_fn = nn.CrossEntropyLoss()
        predictions = self.mlp(pooled)  # Assuming self.mlp is defined in the class
        ### TODO: apply normalisation to prediction (softmax?)

        # predictions = self.cls_head(embedding_n)  # Classifier head for final output
        # print(f"Predictions shape: {predictions.shape}")
        # Average pool over the 16 (num_patches) dimension to get shape (batch_size, dim_out)
        # print(f"Pooled predictions shape: {pooled.shape}")

        # Compute cross-entropy loss
        return F.cross_entropy(predictions, target_labels)
        


        # target_labels should be class indices (LongTensor), not one-hot encoded
        # return F.cross_entropy(predictions, target_labels)

class EncodingBlock(torch.nn.Module):
    def __init__(self, dim_in=49, dim_proj=49, dim_out=49, num_heads=8):
        super().__init__()

        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(dim_in=dim_in, dim_proj=dim_in, dim_out=dim_out) for _ in range(num_heads)
        ])
        self.W_out_proj = torch.nn.Linear(392, 49)  # Linear projection after concatenation of attention heads
        # self.mlp = MLP(input_dim=49, hidden_dim=25, output_dim=10)  # Example MLP for classification
        # print(f"Shape of out_proj weight: {self.W_out_proj.weight.shape}")
        # assert self.out_proj.weight.shape == (16, 49),  f"Expected out_proj weight shape (16, 49), got {self.out_proj.weight.shape}"

    def forward(self, embedding):
        # image_columns = image_to_patch_columns  # Assuming image is already embedded
        head_outputs = [head(embedding) for head in self.heads]

        ### concat all the outputs of the attention heads
        concat = torch.cat(head_outputs, dim=-1)  # Concatenate outputs of all attention heads along the feature dimension

        ### linear projection of the concatenated output
        # print(f"Shape of concatenated output: {concat.shape}") ## 
        # print(f"Shape of out_proj weight: {self.W_out_proj.weight.shape}")
        out_proj = torch.matmul(concat, self.W_out_proj.weight.t())  # Equivalent to self.W_out_proj(concat) without bias
        # print(f"Shape of output after projection: {out_proj.shape}")
        return out_proj  # Return the projected output
        # return concat @ out_proj  # Project the concatenated output to the desired output dimension
    
class SelfAttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_proj, dim_out): ### TODO: seperate dim_v, dim_qk
        ### TODO: include batch size in assertions
        super().__init__()
        self.dim_in = dim_in
        self.dim_proj = dim_proj
        self.dim_out = dim_out  
        self.W_v = torch.nn.Linear(dim_in, dim_proj) ### (49, Y) Y=Z
        self.W_q = torch.nn.Linear(dim_in, dim_proj) ### (49, Z)
        self.W_k = torch.nn.Linear(dim_in, dim_proj) ### (49, Z)
        self.W_h = torch.nn.Linear(dim_proj, dim_out) ### (Y, 49)    
    
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
