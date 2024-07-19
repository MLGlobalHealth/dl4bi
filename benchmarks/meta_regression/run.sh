#!/usr/bin/env bash
python celeba.py +model=sptx_full_256v4 +seed=7 +wandb=True
python celeba.py +model=sptx_fast +seed=7 +wandb=True +name="RFF 256v4"
python celeba.py +model=sptx_full_256v10 +seed=7 +wandb=True
