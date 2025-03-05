import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig:
    def __init__(
        self,
        # Size of the embeddings vector of this Vision Transformer
        hidden_size = 768, 
        # Size of the linear layer for the feedforward network
        intermediate_size = 3072,
        # Number of layers in this Vision Transformer
        num_hidden_layers: int = 12,
        # Number of attention heads
        nums_attention_heads= 12,
        # Number of channels in the input image (e.g., RGB has 3 channels)
        nums_channels: int = 3,
        # Resize the input image to 224 x 224 x 3
        image_size: int = 224,
        # Each image will be divided into patches of 16 x 16
        patch_size: int = 16,
        # Layer normalization epsilon
        layer_norm_eps: float = 1e-6,
        # Dropout rate for attention layers
        attention_dropout: float = 0.0,
        # Number of output embeddings for each image
        num_image_tokens: int = None,
        **kwargs
    ):
        # a call to the initializer method of a parent class
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.nums_attention_heads = nums_attention_heads
        self.nums_channels = nums_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeded_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        """ 
        The patch embedding is extracting information 
        from the input image patch by patch where there is no overlap between the patches.
        """
        self.patch_embedding = nn.Conv2d(
            in_channels=config.nums_channels,
            out_channels=self.embeded_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"# "valid" means no padding is added
        )
        
        # how many patches are there in an image 224 x 224 image with 16 x 16 patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Define the position embeddings
        # need to encode information about the where the patch is located in the image
        self.num_positions = self.num_patches
        # vector of same size as the embeddings vector
        self.position_embeddings = nn.Embedding(self.num_positions, self.embeded_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.nums_positions).expand((1, -1)),
            persistent=False
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # Nonlinear hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tahn")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = nn.SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # Layer normalization: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # self-attention: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        # Get the contextualized embeddings from the self-attention mechanism
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        #[Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # layer normalization: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # MLP: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        # transform it independently from each other
        # Using MLP making the model more degree of freedom to learn the patterns in the data
        # prepare the sequence of patches for the next layer
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        # Extract patches from the input image and convert them to embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        # Transformer encoder layers
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    # Forward pass through the Vision Transformer
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Convert images to embeddings by extracting patches
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
class SiglipVisionModel(nn.Module):
    # config is an instance of SiglipVisionConfig, which contains the parameters for the Vision Transformer model
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    # pixel_values is loaded from numpy array
    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channel, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
        