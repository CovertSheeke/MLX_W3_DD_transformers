import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from patch_and_embed import image_to_patch_columns

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
    def __init__(self, dim_in=49, dim_proj=49, dim_out=49, num_heads=8, num_encoders=6):
        super().__init__()

        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(dim_in=49, dim_proj=49, dim_out=49, num_heads=8) for _ in range(num_encoders)
        ])

        self.mlp = MLP(input_dim=49, hidden_dim=25, output_dim=10)  # Example MLP for classification
        # self.cls_head = nn.Linear(dim_out, 10)  # Classifier head for final output
      
    def forward(self, embedding, target_labels):
        embedding_n = embedding
        for encoding_block in self.encoding_blocks:
            embedding_n = encoding_block(embedding_n)

        # print(f"Final embedding_n shape before loss: {embedding_n.shape}")
        # print(f"Target labels shape: {target_labels.shape}")

        pooled = embedding_n.mean(dim=1)
        # loss_fn = nn.CrossEntropyLoss()
        predictions = self.mlp(pooled)  # Assuming self.mlp is defined in the class

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

    def another_method(self):
        print("This is another method in the EncodingBlock class.")

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
    def __init__(self, dim_in, dim_proj, dim_out):
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

        eps = 1e-6  # Small epsilon to avoid division by zero
        scale = np.sqrt(K.size(-1)) + eps
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
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
