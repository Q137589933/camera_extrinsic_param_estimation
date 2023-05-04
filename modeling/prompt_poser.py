"""
    the main architecture of PromptPoser
"""
from typing import Tuple, Dict, Union, List

import torch
from torch import nn

from modeling.body.fpn.transformer_encoder_fpn import ImageFPN
from modeling.image_encoder import Backbone
from modeling.modeling_utils import configurable
from modeling.prompt_encoder.text_encoder import LanguageEncoder


class PromptPoser(nn.Module):
    @configurable
    def __init__(self,
                 image_encoder: Backbone,
                 text_encoder: LanguageEncoder,
                 fpn: ImageFPN,
                 pixel_mean: Tuple[float],
                 pixel_std: Tuple[float],
                 device: Union[str, torch.device],
                 num_queries: int = 2
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # record pixel_mean and pixel_std in state_dict
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def extract_visual_feature(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        :param images: N C H W
        :return: extracted multi-scale images feature by image fpn
        [[B,C,H1,W1],[B,C,H2,W2]...]
        """

        images = images.to(self.device)
        # images normalized to [0,1] scale w.r.t pixel_mean and pixel_std
        images = (images - self.pixel_mean) / self.pixel_std
        images = self.image_encoder(images)
        return images

    def extract_text_feature(self, text: List[str]):
        """
        :param text: a batch of text in List. like ['a photo of bike','an image of car'...]
        :return: batch text feature tensor. [B, C, max_len]
        """
        return self.text_encoder.get_text_token_embeddings(text, token=False, norm=True)

    def forward(self, batched_input: Dict):
        """
        :param batched_input: dict of a batch.
            Be like {"images":[img1,img2...,img_bs],"text":[text1,text2,...,text_bs]...}
        :return:
        """
        images = batched_input['images']
        images = self.extract_visual_feature(images)

        text = batched_input['text']
        text = self.extract_text_feature(text)
