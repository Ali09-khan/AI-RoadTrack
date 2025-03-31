#!/bin/bash
cd /home/alikhan/Desktop/projects/project/service_kickoff

# Start TorchServe with container-relative paths
torchserve --start \
  --ts-config config.properties \
  --model-store /home/alikhan/Desktop/projects/sergek_project_1/service_holder \
  --models laneLpService.mar,carDetectService.mar \
  --disable-token-auth \
  --ncs
