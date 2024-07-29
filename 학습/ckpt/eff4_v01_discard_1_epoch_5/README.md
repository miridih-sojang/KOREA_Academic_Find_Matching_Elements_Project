---
license: apache-2.0
base_model: google/efficientnet-b4
tags:
- generated_from_trainer
datasets:
- /mnt/raid6/dltmddbs100/miricanbus/train/train_v01_discard_text_1
model-index:
- name: eff4_v01_discard:1_epoch:5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# eff4_v01_discard:1_epoch:5

This model is a fine-tuned version of [google/efficientnet-b4](https://huggingface.co/google/efficientnet-b4) on the /mnt/raid6/dltmddbs100/miricanbus/train/train_v01_discard_text_1 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.5844

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
| 2.5128        | 0.5   | 182  | 2.5717          |
| 1.9443        | 1.0   | 364  | 2.1737          |
| 1.654         | 1.5   | 546  | 2.1157          |
| 1.5367        | 2.0   | 728  | 1.9545          |
| 1.289         | 2.5   | 910  | 1.9168          |
| 1.2996        | 3.0   | 1092 | 1.8338          |
| 1.1344        | 3.5   | 1274 | 1.8507          |
| 1.0844        | 4.0   | 1456 | 1.8318          |
| 0.9347        | 4.5   | 1638 | 1.6875          |
| 0.9236        | 5.0   | 1820 | 1.7456          |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.3.0
- Tokenizers 0.19.1
