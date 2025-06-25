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
    def __init__(self, config, dim_in=49, dim_proj=49, dim_out=49, num_heads=8):
        super().__init__()
        # load config, config includes all hyperparameters of the run ie dimensions, batch size, number of patches, etc.
        self.config = config
        # --- Test case: print cross-entropy for random predictions and target labels ---
        # Simulate random predictions and target labels for demonstration
        # batch_size = self.config.batch_size
        # num_classes = 10
        # torch.manual_seed(42)
        # random_logits = torch.randn(batch_size, num_classes)
        # random_targets = torch.randint(0, num_classes, (batch_size,))
        # test_loss = F.cross_entropy(random_logits, random_targets)
        # print("Random logits:\n", random_logits)
        # print("Random target labels:\n", random_targets)
        # print("Cross-entropy loss for random predictions:", test_loss.item())
        
        # # --- Test case: perfect predictions (one-hot at correct class) ---
        # perfect_logits = torch.zeros(batch_size, num_classes)
        # perfect_logits[torch.arange(batch_size), random_targets] = 100.0  # Set correct class to 1
        # perfect_loss = F.cross_entropy(perfect_logits, random_targets)
        # # print("Perfect logits:\n", perfect_logits)
        # # Print perfect logits and random targets side by side for comparison
        # for i in range(batch_size):
        #     print(f"Perfect logits[{i}]: {perfect_logits[i].tolist()} | Random target: {random_targets[i].item()}")
        # print("Cross-entropy loss for perfect predictions:", perfect_loss.item())

        # initialise the encoding blocks
        self.encoding_blocks = torch.nn.ModuleList([
            EncodingBlock(self.config, dim_in=49, dim_proj=49, dim_out=49, num_heads=self.config.num_heads) for _ in range(self.config.num_encoders)
        ])

        # initialise the MLPs
        self.cls_head = MLP(input_dim=49, hidden_dim=25, output_dim=10)  # MLP for classification
        self.mlp_between_blocks = MLP(input_dim=49, hidden_dim=49, output_dim=49)  # MLP to apply between encoding blocks
      
    def forward(self, embedding, target_labels):
        embedding_n = embedding
        for encoding_block in self.encoding_blocks:
            embedding_n = encoding_block(embedding_n) # B, num_patches, dim_proj_V
            assert embedding_n.shape[-2:] == (self.config.num_patches, self.config.dim_proj_V), f"Expected embedding_n shape ({self.config.batch_size}, {self.config.num_patches}, {self.config.dim_proj_V}), got {embedding_n.shape}"
            embedding_n = self.mlp_between_blocks(embedding_n) # B, num_patches, dim_out
            assert embedding_n.shape[-2:] == (self.config.num_patches, self.config.dim_out), f"Expected embedding_n shape ({self.batch_size}, {self.config.num_patches}, {self.config.dim_out}), got {embedding_n.shape}"

        pooled = embedding_n.mean(dim=1) # Average pooling over the num_patches dimension: B, dim_out
        assert pooled.shape[-1:] == torch.Size([self.config.dim_out]), f"Expected pooled shape ({self.config.batch_size}, {self.config.dim_out}), got {pooled.shape}"
        predictions = self.cls_head(pooled)  # Assuming self.mlp is defined in the class
        assert predictions.shape[-1:] == torch.Size([10]), f"Expected predictions shape ({self.config.batch_size}, 10), got {predictions.shape}"
        
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == target_labels).float().sum()
        accuracy = correct / predictions.shape[0]
        # print(f"Batch accuracy: {accuracy.item():.4f}")


        # predictions = self.cls_head(embedding_n)  # Classifier head for final output
        # print(f"Predictions shape: {predictions.shape}")
        # # Average pool over the 16 (num_patches) dimension to get shape (batch_size, dim_out)
        # print(f"Pooled predictions shape: {pooled.shape}")
        

        # Compute cross-entropy loss
        return F.cross_entropy(predictions, target_labels), accuracy
        


        # target_labels should be class indices (LongTensor), not one-hot encoded
        # return F.cross_entropy(predictions, target_labels)

class EncodingBlock(torch.nn.Module):
    def __init__(self, config, dim_in=49, dim_proj=49, dim_out=49, num_heads=8):
        super().__init__()
        self.config = config
        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(self.config, dim_in=dim_in, dim_proj=dim_in, dim_out=dim_out) for _ in range(num_heads)
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
    def __init__(self, config, dim_in, dim_proj, dim_out): ### TODO: seperate dim_v, dim_qk
        ### TODO: include batch size in assertions
        super().__init__()
        self.config = config
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
