# Copyright (C) 2022 Intel Corporation
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
# http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
 

# SPDX-License-Identifier: Apache-2.0

import argparse, os
from urllib.parse import uses_fragment
import pandas as pd
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizerFast
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
from transformers import Trainer
from transformers import TrainingArguments
from neural_compressor.experimental import Quantization, common
from neural_compressor.metric import METRICS
from transformers import AdamW
from torch.nn import functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from neural_compressor.experimental import Quantization, common
import time

import pdb

# Apply Transfer Learning/Fine-tuning on HuggingFace model
def train_hf_bert(output_model_path):
    def get_datasets():
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        train_dataset, test_dataset = train_dataset.map(tokenize_function, batched=True), test_dataset.map(tokenize_function, batched=True)
        train_dataset, test_dataset = train_dataset.rename_column("label", "labels"), test_dataset.rename_column("label", "labels") #MRPC specific

        train_dataset.set_format(type='torch')
        test_dataset.set_format(type='torch')

        return train_dataset, test_dataset

    # initialization
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.train()
 
    #train_dataloader, test_dataloader = get_datasets()
    train_dataset, test_dataset = get_datasets()

    #init
    def compute_metrics_acc(model_output):
        acc_metric = load_metric("accuracy")
        logits, labels = model_output
        return acc_metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    #if args.use_distributed_training == 'Y':
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    train_args = TrainingArguments(
        output_dir="./outputs/trained_model",
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        use_ipex=True,
        #bf16=False,
        no_cuda=True,
        local_rank=rank   #Use os.environ["RANK"] for CPU trainings
        #xpu_backend='mpi'
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_acc
    )   

    #Training and eval
    trainer.train()
    trainer.save_model()
    return


if __name__ == '__main__':
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    
    args, _ = parser.parse_known_args()
    dist.init_process_group("gloo")

    #Perform training
    start = time.time()
    train_hf_bert(os.path.join('.', 'HF_BertBase_MRPC'))
    end = time.time()
    print("Time for training: " + str(end - start) + "s")