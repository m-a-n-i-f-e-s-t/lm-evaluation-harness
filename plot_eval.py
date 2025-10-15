from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
import argparse
import subprocess
import sys
from collections import defaultdict
from typing import List, Any, Annotated
from pydantic import BaseModel, model_validator, Field
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
import re


EVAL_BASE_DIR = Path("/storage/evals/lm_eval/")

class Metric(BaseModel):
    filter: str
    metric: str
    value: float
    stderr: float

class TaskResult(BaseModel):
    task: str
    alias: str
    metrics: List[Metric]

class EvalResult(BaseModel):
    configs: dict[str, Any]
    versions: dict[str, int]
    n_shot: Annotated[dict[str, int], Field(alias="n-shot")]
    higher_is_better: dict[str, dict[str, bool]]
    n_samples: Annotated[dict[str, int | dict], Field(alias="n-samples")]
    config: dict[str, Any]
    results: dict[str, Any]
    group_subtasks: dict[str, Any]
    groups: dict[str, Any]
    start_time: float
    end_time: float
    total_evaluation_time_seconds: float
    model: Annotated[str, Field(default="")]
    tasks: Annotated[List[TaskResult], Field(default_factory=list)]
    batch_size: int = Field(default=0)
    limit: int = Field(default=99999999)
    step: int = Field(default=0)
    
    @model_validator(mode="after")
    def validate_results(self) -> "EvalResult":
        args = self.config["model_args"]
        if isinstance(args, str):
            args = args.split(',')
            for arg in args:
                if arg.startswith('pretrained='):
                    self.model = arg.split('=')[1]
                    break
        elif isinstance(args, dict):
            self.model = args["pretrained"]
        step_match = re.search(r"step-(\d+)", self.model)
        if step_match:
            self.step = int(step_match.group(1))
            self.model = self.model.replace(f"step-{self.step}", "")
        if not self.model:
            raise ValueError("Model not found in model_args")
        self.limit = self.config.get("limit", 99999999)
        self.batch_size = self.config["batch_size"]
        for task in self.results:
            metric_keys = [key for key in self.results[task] if key != 'alias']
            metrics = []
            metric_filters = list(set([m.replace('_stderr', '') for m in metric_keys]))
            for metric_filter in metric_filters:
                name = metric_filter.split(',')[0]
                filter = metric_filter.split(',')[1]
                metric_value = self.results[task][metric_filter]
                metric_stderr = self.results[task][f"{name}_stderr,{filter}"]
                metric = Metric(filter=filter, metric=name, value=metric_value, stderr=metric_stderr)
                metrics.append(metric)
            task_result = TaskResult(task=task, alias=self.results[task]['alias'], metrics=metrics)
            self.tasks.append(task_result)
        self.results = {}
        return self


def load_results(eval_dir: Path) -> List[EvalResult]:
    results = []
    for result_file in eval_dir.glob("*/**/results_*.json"):
        with open(result_file, "r") as f:
            r = json.load(f)
            result = EvalResult(**r)
            results.append(result)
    return results


def exclude_models(results: List[EvalResult], exclude_models: List[str]) -> List[EvalResult]:
    """Exclude models from the results"""
    excluded_models = [r.model for r in results if any(exclude_model in r.model for exclude_model in exclude_models)]
    print(f"Excluded models: {excluded_models}")
    return [r for r in results if r.model not in excluded_models]


def dedup_results(results: List[EvalResult]) -> List[EvalResult]:
    """Deduplicate results by model and step, keeping the eval results with largest limit and latest timestamp"""
    results_dict = {}
    for r in results:
        key = (r.model, r.step)
        if key not in results_dict or (results_dict[key].limit <= r.limit and results_dict[key].end_time < r.end_time):
            results_dict[key] = r
    return list(results_dict.values())


def plot_results(results: List[EvalResult], output_path: str, include_subtasks: bool, baseline: str = None):
    """plot the results"""
    results_dict = defaultdict(dict)
    for r in results:
        results_dict[r.model][r.step] = r

    subtasks = []
    for r in results:
        for _, subs in r.group_subtasks.items():
            subtasks.extend(subs)
    subtasks = list(set(subtasks))
    print(f"Subtasks: {subtasks}")
    
    metrics_data = defaultdict(lambda: defaultdict(lambda: {'steps': [], 'values': [], 'stderrs': []}))
    baseline_metrics = {}
    baseline_model = None
    
    for model_name, model_results in results_dict.items():
        is_baseline = baseline and baseline in model_name.lower()
        
        for step, result in sorted(model_results.items()):
            for task in result.tasks:
                if not include_subtasks and task.task in subtasks:
                    continue
                for metric in task.metrics:
                    key = f"{task.task}\nmetric:{metric.metric},filter:{metric.filter}"
                    
                    if is_baseline:
                        if not baseline_model:
                            baseline_model = model_name
                        if key not in baseline_metrics:
                            baseline_metrics[key] = metric.value
                    else:
                        metrics_data[key][model_name]['steps'].append(step)
                        metrics_data[key][model_name]['values'].append(metric.value)
                        metrics_data[key][model_name]['stderrs'].append(metric.stderr)
    
    if baseline and baseline_model:
        print(f"Using baseline model: {baseline_model} with {len(baseline_metrics)} metrics")
    
    n_metrics = len(metrics_data)
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_name, models_data) in enumerate(sorted(metrics_data.items())):
        ax = axes[idx]
        
        for model_name, data in sorted(models_data.items()):
            steps = data['steps']
            values = data['values']
            stderrs = data['stderrs']
            
            line = ax.plot(steps, values, marker='o', label=model_name, linewidth=2, markersize=4)
            color = line[0].get_color()
            
            lower = [v - s for v, s in zip(values, stderrs)]
            upper = [v + s for v, s in zip(values, stderrs)]
            ax.fill_between(steps, lower, upper, alpha=0.2, color=color)
        
        if metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name]
            ax.axhline(y=baseline_value, color='#FF6B9D', linestyle='--', linewidth=2, label=f'Baseline: {baseline}', alpha=0.8)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(metric_name, fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation results, this program will go through the eval_dir recursively and find all results_*.json files and plot the results")
    parser.add_argument("--eval_dir", type=str, help="Directory containing the evaluation results", default=EVAL_BASE_DIR, required=False)
    parser.add_argument("--output_path", type=str, help="Path to save the plots", default=None, required=False)
    parser.add_argument("--include_subtasks", action="store_true", help="Include subtasks in the plot", default=False, required=False)
    parser.add_argument("-x", "--exclude_models", type=str, nargs="*", help="Model regex to exclude from plotting", default=[], required=False)
    parser.add_argument("-b", "--baseline", type=str, help="Baseline model to plot", default=None, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    results = load_results(eval_dir)
    results = exclude_models(results, args.exclude_models)
    results = dedup_results(results)
    plot_results(results, args.output_path, args.include_subtasks, args.baseline.lower() if args.baseline else None)

if __name__ == "__main__":
    main()
    