# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch EfficientNet model."""

from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from dataclasses import dataclass

from transformers import EfficientNetPreTrainedModel
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetEmbeddings, EfficientNetConfig, EfficientNetEncoder
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/efficientnet-b7"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/efficientnet-b7"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


EFFICIENTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@dataclass
class EffOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_removed: torch.FloatTensor = None
    logits_per_elements: torch.FloatTensor = None
    elements_embeds: torch.FloatTensor = None
    removed_embeds: torch.FloatTensor = None
    elements_model_output: BaseModelOutputWithPooling = None
    removed_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["elements_model_output", "removed_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )



def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@add_start_docstrings(
    "The bare EfficientNet model outputting raw features without any specific head on top.",
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetDualModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = EfficientNetEmbeddings(config)
        self.encoder = EfficientNetEncoder(config)

        # Final pooling layer
        if config.pooling_type == "mean":
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")
        
        self.vision_embed_dim = config.hidden_dim
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=EffOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        elements_pixel_values: torch.FloatTensor = None,
        removed_pixel_values: torch.FloatTensor = None,
        return_loss: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EffOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if elements_pixel_values is None:
            raise ValueError("You have to specify elements_pixel_values")
        if removed_pixel_values is None:
            raise ValueError("You have to specify removed_pixel_values")

        elements_embedding_output = self.embeddings(elements_pixel_values)
        removed_embedding_output = self.embeddings(removed_pixel_values)

        elements_vision_outputs = self.encoder(
            elements_embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        removed_vision_outputs = self.encoder(
            removed_embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply pooling
        elements_last_hidden_state = elements_vision_outputs[0]
        elements_pooled_output = self.pooler(elements_last_hidden_state)
        # Reshape (batch_size, 1792, 1 , 1) -> (batch_size, 1792)
        elements_pooled_output = elements_pooled_output.reshape(elements_pooled_output.shape[:2])

        elements_image_embeds = self.visual_projection(elements_pooled_output)


        # Apply pooling
        removed_last_hidden_state = removed_vision_outputs[0]
        removed_pooled_output = self.pooler(removed_last_hidden_state)
        # Reshape (batch_size, 1792, 1 , 1) -> (batch_size, 1792)
        removed_pooled_output = removed_pooled_output.reshape(removed_pooled_output.shape[:2])

        removed_image_embeds = self.visual_projection(removed_pooled_output)


        # normalized features
        elements_image_embeds = elements_image_embeds / elements_image_embeds.norm(p=2, dim=-1, keepdim=True)
        removed_image_embeds = removed_image_embeds / removed_image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_elements = torch.matmul(elements_image_embeds, removed_image_embeds.t().to(elements_image_embeds.device)) * logit_scale.to(
            elements_image_embeds.device
        )
        logits_per_removed = logits_per_elements.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_elements)

        if not return_dict:
            output = (logits_per_removed, logits_per_elements, elements_image_embeds, removed_image_embeds, elements_vision_outputs, removed_vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return EffOutput(
            loss=loss,
            logits_per_removed=logits_per_removed,
            logits_per_elements=logits_per_elements,
            elements_embeds=elements_image_embeds,
            removed_embeds=removed_image_embeds,
            elements_model_output=elements_vision_outputs,
            removed_model_output=removed_vision_outputs,
        )