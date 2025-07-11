## PROJECT NOT UNDER ACTIVE MANAGEMENT

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  

Contact: webadmin@linux.intel.com
# **Intel® NLP workflow for Azure** ML - Inference

## Overview
This workflow demonstrates how users can utilize Intel’s XPU hardware (e.g.: CPU - Ice Lake or above) and related optimized software to perform distributed training and inference on the Azure Machine Learning Platform. The main software packages used here are Intel Extension for PyTorch, PyTorch, HuggingFace, Azure Machine Learning Platform, and Intel Neural Compressor. For more detailed information, please visit the [Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) GitHub repository.

## How it Works
This workflow utilizes the infrastructure provided by AzureML.

### Architecture 

AzureML:

![azureml_architecture](https://user-images.githubusercontent.com/43555799/205149722-e37dcec5-5ef2-4440-92f2-9dc243b9e556.jpg)

### Model Spec
The uncased BERT base model is used to demonstrate this workflow.

```python
bert-base-uncased-config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 128,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.21.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

### Dataset 
Microsoft Research Paraphrase Corpus is used as the dataset for training and testing. 

| **Type**                 | **Format** | **Rows** 
| :---                     | :---       | :---     
| Training Dataset         | HuggingFace Dataset  | 3668
| Testing  Dataset         | HuggingFace Dataset  | 1725

## Get Started

### **Prerequisites**
Docker is required to start this workflow. You will also need Azure credentials to perform any training/inference related operations.

For setting up the Azure Machine Learning Account, you may refer to the following link:
<br>
https://azure.microsoft.com/en-us/free/machine-learning

For configuring the Azure credentials using the Command-Line Interface, you may refer to the following link:
<br>
https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli

The following two websites list out the availability and type of the instances for users. Users may choose the appropriate instances based on their needs and region:
<br>
https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target
<br>
https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east

#### Download the Repo
Clone [Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) repository into your working directory.
```
git clone https://github.com/intel/Intel-NLP-workflow-for-Azure-ML.git .
git checkout v1.0.1
```

#### Download the Datasets
The dataset will be downloaded the first time the training runs.

### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image to run the entire pipeline. 
For interactive set up, please go to [Interactive Docker](#interactive-docker).

#### Setup 
Download the `config.json` file from your Azure ML Studio Workspace.

##### Pull Docker Image
```
docker pull intel/ai-workflows:nlp-azure-inference
```

#### How to Run 
Use the inference script `1.0-intel-azureml-inference.py` and downloaded `config.json` file to run the inference pipeline.

The code snippet below runs the inference session. This session will call the FP32 model generated during the training session from the `notebooks/fp32_model_output` folder.
```
export AZURE_CONFIG_FILE=<path to config file downloaded from Azure ML Studio Workspace>

docker run \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${PWD}/notebooks:/root/notebooks \
  --volume ${PWD}/src:/root/src \
  --volume ${PWD}/${AZURE_CONFIG_FILE}:/root/notebooks/config.json \
  --workdir /root/notebooks \
  --privileged --init -it \
  intel/ai-workflows:nlp-azure-inference \
  sh -c "jupyter nbconvert --to python 1.0-intel-azureml-inference.ipynb && python3 1.0-intel-azureml-inference.py"
```

### **Interactive Docker**
Below setup and how-to-run sessions are for users who want to use an interactive environment.  
For docker pipeline, please go to [docker session](#docker).
#### Setup 

Build the docker image to prepare the environment for running the Jupyter notebooks.
```
cd scripts
sh build_main_image.sh
```

Use the built docker image (by `build_main_image.sh`) to run the Jupyter notebooks. Execute the following command:
```bash
sh start_script.sh
```
After starting the container, execute the following command in the interactive shell.
```bash
cd notebooks
jupyter notebook --allow-root
```
Start the notebook that is named as inference. Set number of physical cores in score_hf.py according to the machine. The variable should contain a space-separated or comma-separated list of physical CPUs. The best configuration found for Standard_D16_v5 is currently set as default, but users may choose to explore different numbers of physical cores for different machines.

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation. 
| **Name**         | **Description**                                           |
| ----------       | ----------------------------------------------------------|
| CPU              | 4th Generation Intel® Xeon® Scalable Processors           |
| Usable RAM       | 16 GB                                                     |
| Disk Size        | 256 GB                                                    |

## Useful Resources 
[Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
<br>
[Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Support Forum 
[Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) tracks both bugs and enhancement requests using GitHub. We welcome input, however, before filing a request, please make sure you do the following: Search the GitHub issue database.
