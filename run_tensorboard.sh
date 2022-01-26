#!/usr/bin/env bash

docker run \
  -v $(pwd)/src:/code \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  -v $(pwd)/models:/models \
  -p 6006:6006 \
  --gpus '"device=1"' \
  -it icgan /bin/bash