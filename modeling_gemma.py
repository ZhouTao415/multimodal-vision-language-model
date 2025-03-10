import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

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
        