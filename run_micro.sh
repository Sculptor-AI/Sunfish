#!/bin/bash
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

venv/bin/python train.py --config micro --cpu --name micro-coherent-gen "$@"
