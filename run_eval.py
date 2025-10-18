from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
import argparse
import subprocess
import sys
from typing import List
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import logging
import json


EVAL_BASE_DIR = Path("/storage/evals/lm_eval/")

@dataclass
class EvalConfig:
    model: str
    tasks: List[str]
    batch_size: int
    tp_size: int
    limit: int
    log_samples: bool

def run_eval(config: EvalConfig) -> dict | None:
    """
    Run an evaluation for a given configuration.
    """

    if config.log_samples:
        if config.model.startswith("/"):
            model_name = config.model.split("/")[-1]
        else:
            model_name = config.model
        output_dir = EVAL_BASE_DIR / model_name.replace("/", "_")
        evaluation_tracker = EvaluationTracker(output_path=output_dir)
    else:
        evaluation_tracker = None

    results = simple_evaluate(
        model="vllm",
        model_args={
            "pretrained": config.model,
            "enable_prefix_caching": False,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": config.tp_size,
        } | ({"distributed_executor_backend": "mp", "dtype": "auto"} if config.tp_size > 1 else {}),
        tasks=config.tasks, 
        batch_size=config.batch_size,
        limit=config.limit,
        log_samples=config.log_samples,
        evaluation_tracker=evaluation_tracker,
    )

    if results is not None:
        if config.log_samples:
            samples = results.pop("samples")
    
        if evaluation_tracker is not None:
            evaluation_tracker.save_results_aggregated(results=results, samples=samples if config.log_samples else None)

        if config.log_samples and evaluation_tracker is not None:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    return results
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for a given checkpoint directory or a directory containing many checkpoint subdirectories, results will be saved in the same directory as the checkpoint directory in a results_<timestamp>.json file")
    parser.add_argument("model", type=str, help="model to be evaluated, can be a hf name, a path to a checkpoint directory, or a path to a directory containing many checkpoint subdirectories")
    parser.add_argument("--tasks", type=str, required=False, default=["hendrycks_math", "hellaswag", "piqa", "arc_challenge", "gsm8k_cot", "mmlu", "winogrande", "arc_all_options"], help="Tasks to evaluate", nargs="+")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size to use for evaluation")
    parser.add_argument("--log_samples", action="store_true", required=False, default=True, help="Whether to log samples")
    parser.add_argument("--limit", type=int, required=False, default=None, help="Limit the number of examples per task to evaluate")
    parser.add_argument("--tp_size", type=int, required=False, default=1, help="TP size to use for evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    def run_eval_for_path(model: str):
        config = EvalConfig(model=model, tasks=args.tasks, batch_size=args.batch_size, log_samples=args.log_samples, limit=args.limit, tp_size=args.tp_size)
        res = run_eval(config)
        return res

    if not Path(args.model).exists(): # model is a hf name
        logging.info(f"{args.model} seems to be a hf name, running eval")
        res = run_eval_for_path(args.model)
        logging.info(f"Evaluated {args.model}")
        print(make_table(res))
    else:
        path = Path(args.model)
        if path.exists() and (path / "config.json").exists():
            logging.info(f"{args.model} seems to be a path to a checkpoint directory, running eval")
            logging.info(f"Evaluating {path}")
            res = run_eval_for_path(args.model)
            logging.info(f"Evaluated {args.model}")
            print(make_table(res))
        else:
            logging.info(f"{args.model} seems to be a path to a directory containing many checkpoint subdirectories, running eval on each checkpoint directory")
            for checkpoint_dir in path.iterdir():
                if not checkpoint_dir.is_dir() or (checkpoint_dir / "config.json").exists():
                    logging.info(f"{checkpoint_dir} is not a valid checkpoint directory, skipping")
                    continue
                logging.info(f"Evaluating {checkpoint_dir}")
                res = run_eval_for_path(str(checkpoint_dir))
                logging.info(f"Evaluated {str(checkpoint_dir)}")
                print(make_table(res))
