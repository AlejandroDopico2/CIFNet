import argparse
import datetime
from typing import Any, Dict
import torch


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    experiment_name = datetime.datetime.now().strftime("experiment_%Y-%m-%d_%H-%M-%S")

    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "name": experiment_name,
        "binary": args.binary,
        "backbone": args.backbone,
        "rolann_lamb": args.rolann_lamb,
        "dataset": args.dataset,
        "pretrained": args.pretrained,
        "reset": args.reset,
        "use_wandb": args.use_wandb,
        "device": device,
        "flatten": False if args.backbone else True,
        "samples_per_task": args.samples_per_task,
        "sparse": args.sparse,
        "dropout_rate": args.dropout_rate,
        "initial_tasks": args.initial_tasks,
        "num_tasks": args.num_tasks,
        "classes_per_task": args.classes_per_task,
        "freeze": args.freeze,
        "buffer_size": args.buffer_size,
    }

    return config
