---
license: apache-2.0
base_model: google/efficientnet-b4
tags:
- generated_from_trainer
datasets:
- /mnt/raid6/dltmddbs100/miricanbus/train/train_v01
model-index:
- name: eff4_v01_epoch:5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# eff4_v01_epoch:5

This model is a fine-tuned version of [google/efficientnet-b4](https://huggingface.co/google/efficientnet-b4) on the /mnt/raid6/dltmddbs100/miricanbus/train/train_v01 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8764

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 4
- total_train_batch_size: 1024
- total_eval_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant_with_warmup
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 2.4401        | 0.5   | 395  | 2.6278          |
| 1.8281        | 1.0   | 790  | 2.3536          |
| 1.607         | 1.5   | 1185 | 2.2261          |
| 1.4997        | 2.0   | 1580 | 2.1249          |
| 1.2472        | 2.5   | 1975 | 2.0170          |
| 1.2239        | 3.0   | 2370 | 2.0372          |
| 1.0766        | 3.5   | 2765 | 1.9693          |
| 0.9488        | 4.0   | 3160 | 1.9368          |
| 0.9594        | 4.5   | 3555 | 1.9754          |
| 0.9399        | 5.0   | 3950 | 1.9116          |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.3.0
- Tokenizers 0.19.1
