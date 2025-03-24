import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from utils import save_config, load_config
from evaluation import evalrank
from co_train import main

# current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + str(random.randint(0, 100))
current_time = "2025-03-16-13-05"

# Hyper Parameters
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument(
    "--data_path", default="/home/lzy/data/f30k", help="/home/lzy/data/f30k"
)
parser.add_argument(
    "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k}_precomp"
)
parser.add_argument(
    "--vocab_path",
    default="/home/lzy/data/vocab",
    help="Path to saved vocabulary json files.",
)

parser.add_argument(
    "--img_dim",
    default=2048,
    type=int,
    help="Dimensionality of the image embedding.",
)
parser.add_argument(
    "--word_dim",
    default=300,
    type=int,
    help="Dimensionality of the word embedding.",
)
parser.add_argument(
    "--embed_size",
    default=1024,
    type=int,
    help="Dimensionality of the joint embedding.",
)
parser.add_argument(
    "--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding."
)
parser.add_argument(
    "--num_layers", default=1, type=int, help="Number of GRU layers."
)
parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
parser.add_argument(
    "--no_imgnorm",
    action="store_true",
    help="Do not normalize the image embeddings.",
)
parser.add_argument(
    "--no_txtnorm",
    action="store_true",
    help="Do not normalize the text embeddings.",
)
parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

# noise settings
parser.add_argument("--noise_file", default="noise_index/f30k_precomp_0.4.npy", help="noise_file")
parser.add_argument("--noise_ratio", default="0.4", type=float, help="Noisy ratio")

# NCR Settings
parser.add_argument(
    "--no_co_training", action="store_true", help="No co-training for noisy label."
)
parser.add_argument("--warmup_epoch", default=10, type=int, help="warm up epochs")
parser.add_argument("--warmup_model_path", default="/home/lzy/paper_code/BiCro-main-base/output/2025_03_19_19_47_1643/train_f30k/checkpoint_dev_best.pth.tar", help="warm up models")
parser.add_argument(
    "--p_threshold", default=0.5, type=float, help="clean probability threshold"
)
parser.add_argument(
    "--soft_margin", default="exponential", help="linear|exponential|sin"
)
parser.add_argument(
    "--noise_train", default="noise_soft", help="noise selection train|noise_soft|noise_hard"
)
parser.add_argument(
    "--noise_tem", default=0.5, type=float, help="noise_soft temperature"
)
parser.add_argument(
    "--warmup_type", default='warmup_sele', help="noise_soft temperature"
)
parser.add_argument(
    "--fit_type", default='bmm', help="gmm bmm"
)
parser.add_argument(
    "--warmup_rate", default=0.5, type=float, help="warmup ratio"
)
# Runing Settings
parser.add_argument("--gpu", default="4", help="Which gpu to use.")
parser.add_argument(
    "--seed", default=42, type=int, help="Random seed."
)
parser.add_argument(
    "--output_dir", default=os.path.join("output", current_time), help="Output dir."
)
parser.add_argument(
    "--saved_model", default='', help="Output dir."
)
parser.add_argument(
    "--id", default="train_f30k", help="run id"
)


opt = parser.parse_args()
opt.output_dir = os.path.join(opt.output_dir + '/', opt.id)
evalrank(os.path.join(opt.output_dir, "checkpoint_dev_best.pth.tar"), split="test")