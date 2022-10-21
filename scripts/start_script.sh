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

#This script will run an interactive environment. Users will run the launcher codes (i.e. the jupyter notebooks) within the container.
#Users may pay attention to the firewall/proxy issue. Users may need to add -e "http_proxy=http://xxxx.com:xxx" -e "https_proxy=http://xxxx.com:xxx" -e "no_proxy=localhost,127.0.0.1" 
docker run -it -v $(pwd)/../notebooks:/root/notebooks -v $(pwd)/../src:/root/src --net=host intel_microsoft_cloud_trainandinf:latest /bin/bash