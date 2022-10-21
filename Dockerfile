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

#The docker for running the Azure ML training and infernece
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install --no-install-recommends curl=7.68.0-1ubuntu2.13 -y && \
    apt-get install --no-install-recommends python3-pip=20.0.2-5ubuntu1.6 -y && \
    rm -r /var/lib/apt/lists/*

# Install software packages for running the notebooks, AzureML infra. and related source codes
RUN pip install --no-cache-dir azureml-sdk==1.45.0 && pip install --no-cache-dir notebook==6.4.12
