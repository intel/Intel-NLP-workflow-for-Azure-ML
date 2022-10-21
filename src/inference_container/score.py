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
import onnxruntime
import sys
import os
import time
from transformers import AutoTokenizer

# Initialization
def init():
    global session, input_name, output_name, tokenizer
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'quantized_bert_base_MRPC_model.onnx')
    session = onnxruntime.InferenceSession(model, None)
    
    #Model dependent
    max_seq_length=128
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding="max_length", max_length=max_seq_length)
    return
    
# The main function for inference
def run(input_data):
    def preprocess(input_data_json):
        # convert the JSON data into the tensor input
        text1 = json.loads(input_data_json)['sentence1']
        text2 = json.loads(input_data_json)['sentence2']
        processed_input = tokenizer(text1, text2, return_tensors="np")
        return processed_input

    def postprocess(result):
        return int(np.argmax(np.array(result).squeeze(), axis=0))

    try:
        input_data = preprocess(input_data)
        result = session.run([], {"input_ids": input_data['input_ids'], "token_type_ids": input_data['token_type_ids'], "attention_mask": input_data['attention_mask']})
        postprocessed_result = postprocess(result)
        result_dict = {"result": postprocessed_result}
    except Exception as e:
        result_dict = {"error": str(e)}
    return result_dict