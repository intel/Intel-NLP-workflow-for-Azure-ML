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

import json
import numpy as np
import sys
import os
import time
from transformers import AutoTokenizer
import torch
from torch.nn import functional as F
from neural_compressor.utils.load_huggingface import OptimizedModel


#Optimizations - depends on the vCPUs of the instance. Please change accordingly.
os.environ['GOMP_CPU_AFFINITY']='0-3'
os.environ['OMP_PROC_BIND']='CLOSE'
os.environ['OMP_SCHEDULE']='STATIC'
torch.set_flush_denormal(True)


# Initialization
def init():
    global model, tokenizer, softmax, model_dir
    model_dir = '/outputs'
    
    model = OptimizedModel.from_pretrained(os.getenv('AZUREML_MODEL_DIR') + model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('AZUREML_MODEL_DIR') + model_dir)
    softmax = torch.nn.Softmax(dim=-1)
    return

# The main function for inference
def run(input_data):
    try:
        sentence1 = json.loads(input_data)['sentence1']
        sentence2 = json.loads(input_data)['sentence2']
        tokenized_inputs = tokenizer(sentence1, sentence2, return_tensors="pt")
        results = model(**tokenized_inputs)
        probability = F.softmax(results['logits'][0], dim=0)
        pred = torch.argmax(probability).item()
        result_dict = {'result' : str(pred), 'sentence1' : str(sentence1), 'sentence2' : str(sentence2), 'logits' : str(results['logits']), 'probability' : str(probability), 'input_data': str(tokenized_inputs), 'model_path' : str(os.getenv('AZUREML_MODEL_DIR') + model_dir)}
    except Exception as e:
        result_dict = {"error": str(e)}

    
    return result_dict