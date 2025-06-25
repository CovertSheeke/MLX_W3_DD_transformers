import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

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

        # initialise the encoding blocks
        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(self.config) for _ in range(self.config.num_encoders)
        ])

        # initialise the MLPs
        self.cls_head = MLP(input_dim=self.config.dim_out, hidden_dim=self.config.mlp_hidden_dim, output_dim=10)  # MLP for classification
        self.mlp_between_blocks = MLP(input_dim=self.config.dim_out, hidden_dim=self.config.mlp_hidden_dim, output_dim=self.config.dim_in)  # MLP to apply between encoding blocks
      
    def forward(self, x, trg):
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
        self.W_out_proj = torch.nn.Linear(self.config.num_heads*self.config.dim_out, self.config.dim_out)  # Linear projection after concatenation of attention heads

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        ### concat all the outputs of the attention heads
        concat = torch.cat(head_outputs, dim=-1)  # Concatenate outputs of all attention heads along the feature dimension
        assert concat.shape[-2:] == (self.config.num_patches, (self.config.num_heads * self.config.dim_out)), f"Expected concatenated output shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.num_heads * self.config.dim_out}), got {concat.shape}"
        ### linear projection of the concatenated output
        out_proj = torch.matmul(concat, self.W_out_proj.weight.t())  # Equivalent to self.W_out_proj(concat) without bias
        assert out_proj.shape[-2:] == (self.config.num_patches, self.config.dim_out), f"Expected output projection shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {out_proj.shape}"
        return out_proj  # Return the projected output
    
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

        # Compute attention scores
        A = Q @ K.transpose(-2, -1) * (K.size(-1) ** -0.5) # (batch_size, num_patches, num_patches)
        assert A.shape[-2:] == (self.num_patches, self.num_patches), f"Expected A shape ({self.num_patches}, {self.num_patches}), got {A.shape}"

        ### TODO: check if this is correct, we were trying to avoid nannvalues, maybe this isn't the best place to do this
        attention_scores = torch.clamp(A, min=-30.0, max=30.0)  # Prevent softmax overflow
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute output
        attention_out = attention_weights @ V  # (batch_size, num_patches, dim_proj)
        assert attention_out.shape[-2:] == (self.num_patches, self.dim_proj_V), f"Expected attention_out shape ({self.num_patches}, {self.dim_proj_V}), got {attention_out.shape}"
        
        # Linear projection for dimensionality
        output = self.W_h(attention_out)  # (batch_size, num_patches, dim_out)
        assert output.shape[-2:] == (self.num_patches, self.dim_out), f"Expected output shape ({self.num_patches}, {self.dim_out}), got {output.shape}"
        

        return output
