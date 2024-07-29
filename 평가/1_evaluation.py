from typing import Any, Optional, Tuple, Union
from image_dataset import ImageDataset
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
    logging,
)
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
import transformers

from tqdm import tqdm

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
    set_seed,
)

from setproctitle import setproctitle
import pandas as pd
setproctitle('miricanvas')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기본 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# 로깅 수준을 WARNING으로 설정
log_level = logging.WARNING
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    logit_scale_init_value: Optional[float] = field(
        default=2.6592,
        metadata={
            "help": (
                "logit_scale_init_value"
            )
        },
    )

@dataclass
class DataArguments:
    csv_file: str = field(
        metadata={"help": "Path to the CSV file containing image paths"}
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "Path to the CSV file containing image paths"}
    )
    image_column: str = field(
        default="image_file_name",
        metadata={"help": "The name of the column in the CSV containing the image file paths."},
    )
    removed_column: str = field(
        default="removed_elem_background_path",
        metadata={"help": "The name of the column in the CSV containing the removed element background image file paths."},
    )
    num_samples: int = field(
        default=1,
        metadata={"help": "Number of random samples to evaluate."}
    )


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def preprocess_image(image, image_processor):
    return image_processor(image, return_tensors="pt")['pixel_values'][0].to(device)

def cosine_similarity(tensor1, tensor2):
    return torch.matmul(tensor1, tensor2.t().to(tensor1.device))

def calculate_mrr(ranks):
    return sum([1.0 / rank for rank in ranks]) / len(ranks)

def calculate_recall_at_k(ranks, k):
    return sum([1 if rank <= k else 0 for rank in ranks]) / len(ranks)

def calculate_recall_at_p(ranks, total_items, p):
    k = int(total_items * p)
    return calculate_recall_at_k(ranks, k)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levellevel)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.WARNING
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model_name = model_args.model_name_or_path.split('/')[-1].replace(':', '_')
    
    output_dir = os.path.join(data_args.output_dir, model_name)
    output_csv_path = os.path.join(output_dir, os.path.splitext(data_args.csv_file)[0].split('_')[-1] + '_with_similarities.csv')
    
    if os.path.exists(output_csv_path):
        logger.warning(f"Output CSV already exists at {output_csv_path}. Skipping execution.")
        return

    df = pd.read_csv(data_args.csv_file)

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config = EfficientNetConfig.from_pretrained(model_args.model_name_or_path)
    config.projection_dim = config.hidden_dim
    config.logit_scale_init_value = model_args.logit_scale_init_value

    model = EfficientNetDualModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    ).to(device)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    set_seed(42)
    total_ranks = []

    num_samples = min(data_args.num_samples, len(df))
    removed_elem_background_paths = df[data_args.removed_column].sample(num_samples).values
    df['y_file_name'] = df.removed_elem_background_path.str.split('/').str[-1]
    miridih_df = ImageDataset()
    print(df.iloc[0].values)
    print(miridih_df)
    exit()
    for removed_elem_background_path in removed_elem_background_paths:
        removed_elem_background_image = load_image(removed_elem_background_path)
        removed_elem_background_image_tensor = preprocess_image(removed_elem_background_image, image_processor).to(device)

        with torch.no_grad():
            removed_embedding_output = model.embeddings(removed_elem_background_image_tensor.unsqueeze(0))
            removed_vision_output = model.encoder(removed_embedding_output)
            removed_last_hidden_state = removed_vision_output[0]
            removed_pooled_output = model.pooler(removed_last_hidden_state).reshape(removed_last_hidden_state.shape[0], -1)
            removed_image_embedding = model.visual_projection(removed_pooled_output)
            removed_image_embedding = removed_image_embedding / removed_image_embedding.norm(p=2, dim=-1, keepdim=True)

        similarities = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            element_image_path = row[data_args.image_column]
            element_image = load_image(element_image_path)
            element_image_tensor = preprocess_image(element_image, image_processor).unsqueeze(0).to(device)

            with torch.no_grad():
                element_embedding_output = model.embeddings(element_image_tensor)
                element_vision_output = model.encoder(element_embedding_output)
                element_last_hidden_state = element_vision_output[0]
                element_pooled_output = model.pooler(element_last_hidden_state).reshape(element_last_hidden_state.shape[0], -1)
                element_image_embedding = model.visual_projection(element_pooled_output)
                element_image_embedding = element_image_embedding / element_image_embedding.norm(p=2, dim=-1, keepdim=True)

            similarity = torch.matmul(element_image_embedding, removed_image_embedding.t().to(element_image_embedding.device)).item()
            similarities.append((element_image_path, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        for correct_element_image_path in df[df[data_args.removed_column] == removed_elem_background_path][data_args.image_column].tolist():
            rank = next(i for i, (path, _) in enumerate(similarities) if path == correct_element_image_path) + 1
            total_ranks.append(rank)

    mrr = calculate_mrr(total_ranks)
    logger.warning(f"MRR: {mrr}")

    total_items = len(df)
    p_values = [0.1, 0.2, 0.25, 0.3, 0.5]
    recall_at_p_values = {}

    for p in p_values:
        recall_at_p = calculate_recall_at_p(total_ranks, total_items, p)
        recall_at_p_values[f'Recall@{int(p*100)}%'] = recall_at_p
        logger.warning(f"Recall@{int(p*100)}%: {recall_at_p}")

    results_df = pd.DataFrame({
        "total_itemts": [total_items],
        "MRR": [mrr],
        **{key: [value] for key, value in recall_at_p_values.items()}
    })

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    logger.warning(f"Saved results to {output_csv_path}")

if __name__ == "__main__":
    main()
    
        