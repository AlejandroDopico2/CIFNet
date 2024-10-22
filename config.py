import argparse
import datetime
from typing import Any, Dict
import torch


def get_batch_config(args: argparse.Namespace) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() and args.backbone else "cpu"
    experiment_name = datetime.datetime.now().strftime("experiment_%Y-%m-%d_%H-%M-%S")

    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "name": experiment_name,
        "backbone": args.backbone,
        "rolann_lamb": args.rolann_lamb,
        "dataset": args.dataset,
        "pretrained": args.pretrained,
        "use_wandb": args.use_wandb,
        "device": device,
        "flatten": False if args.backbone else True,
        "num_instances": args.num_instances,
        "sparse": args.sparse,
        "dropout_rate": args.dropout_rate,
        "num_tasks": args.num_tasks,
        "classes_per_task": args.classes_per_task,
        "freeze_mode": args.freeze_mode,
        "patience": 3,  # TODO: Make an argument.
        "reset": args.reset,
    }

    return config


def get_continual_config(args: argparse.Namespace) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() and args.backbone else "cpu"
    experiment_name = datetime.datetime.now().strftime("experiment_%Y-%m-%d_%H-%M-%S")

    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "name": experiment_name,
        "backbone": args.backbone,
        "rolann_lamb": args.rolann_lamb,
        "dataset": args.dataset,
        "pretrained": args.pretrained,
        "use_wandb": args.use_wandb,
        "device": device,
        "flatten": False if args.backbone else True,
        "samples_per_task": args.samples_per_task,
        "sparse": args.sparse,
        "dropout_rate": args.dropout_rate,
        "initial_tasks": args.initial_tasks,
        "num_tasks": args.num_tasks,
        "classes_per_task": args.classes_per_task,
        "freeze_mode": args.freeze_mode,
        "freeze_rolann": args.freeze_rolann,
        "buffer_size": args.buffer_size,
        "patience": 3,  # TODO: Make an argument.
        "use_eb": args.use_eb,
        "sampling_strategy": args.sampling_strategy,
    }

    return config
