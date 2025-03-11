import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

""" 
        vocab_size, how many tokens we have in the vocabulary
        hidden_size, what is the size of the embeddings vectors of each token
        intermediate_size, wich is the size of the feed forward layer in siglip 
        num_hidden_layers, how many layers we have in the transformer in this  gemma LLM
"""

class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        
""" 
        vision_config: Configurations of the vison encoder
        text_config  : Configurations of the text decoder, which is gemma
        ignore_index=-100,: which are only doing the inference
        image_token_index=256000,: which is the corresponding to the placeholder image token
        vocab_size=257152,
        projection_dim=2048,:what the final dimension that the image features should be resized before fedded to the language model
        hidden_size=2048,: whch is the emebedding size of the LLM, so the LLM has some token, these embeddings have a dim, how many->2048
        pad_token_id=None,
        **kwargs,
"""

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Same how many patches we have in the image or how many token we get in the green block see the picture
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


"""
Main class:
Conditional Generation
-> We are conditioning the generation of the text on the image that is provided as input.
我们根据提供的图像来对文本的生成进行条件约束
"""
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        # Image encoder
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # Convert the size of the embeddings ouput by the vision encoder into the size of the embedding each text token
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple:
        
        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        
        # 1. Extract the input embeddings
        # shape: (Batch_size, Seq_len, Embedding_size)
        # we are converting all the input tokens which are the images tokens 
        # and the beginning of the sentence tokens of the prompt + the new liine character token intot embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 2. Merge the text and images
        # shape: [Btach_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # shape: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)
        
        """ we need tot merge the tokens extracted fromt the vision Model and the tokens extracted from the text embeddings
            which already contain some placeholders for where we should put the images tokens
        """
        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        """ 
        1. We extract first we taking the text, which already contains the placeholders
        2. we replace these placeholders with the features extracted from the vision encoder
        3. we feed everything to the language model
        3. the language model will generate the next token/output
        """
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )