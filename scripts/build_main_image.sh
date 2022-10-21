#!/bin/sh

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

#This script will build the docker image for running the Jupyter notebooks
#Please pay attention to the proxy/firewall settings, users may need to add --build-arg http_proxy=http://xxxx.com:xxx 
docker build -t intel_microsoft_cloud_trainandinf ../