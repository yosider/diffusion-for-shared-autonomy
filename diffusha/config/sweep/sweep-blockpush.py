#!/usr/bin/env python3
import os
import sys

from params_proto.hyper import Sweep

# from diffusha.config import RUN
from diffusha.config.default_args import Args

this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    Args.env_name = "BlockPushMultimodal-v1"
    Args.num_training_steps = 30_000  # Roughly 5 hours per 30k steps
    Args.randp = 0.0

sweep.save(os.path.splitext(this_file_name)[0] + ".jsonl")
