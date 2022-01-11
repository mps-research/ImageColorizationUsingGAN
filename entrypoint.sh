#!/usr/bin/env bash

tensorboard --logdir /logs --bind_all &

exec "$@"
