from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
import argparse
import subprocess
import sys
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Any, Annotated
from pydantic import BaseModel, model_validator, Field
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
import re


EVAL_BASE_DIR = Path("/storage/evals/lm_eval/")
MAX_LIMIT = 99999999

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
    start_time: float
    end_time: float
    total_evaluation_time_seconds: float
    model: Annotated[str, Field(default="")]
    tasks: Annotated[List[TaskResult], Field(default_factory=list)]
    batch_size: int = Field(default=0)
    limit: int = Field(default=MAX_LIMIT)
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
        self.limit = self.config.get("limit", MAX_LIMIT) or MAX_LIMIT
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
        self.model = self.model.lower()
        self.results = {}
        return self


@dataclass
class EvalSummary:
    """ A summary of all eval results
    """
    models: List[str]
    model_steps: dict[str, List[int]]
    tasks: List[str]
    model_task_step_results: dict[str, dict[str, dict[int, TaskResult]]]
    baseline: str
    baseline_task_results: dict[str, TaskResult]
    # Track metadata for deduplication
    _result_metadata: dict[str, dict[str, dict[int, dict]]] = None  # model -> task -> step -> {limit, end_time, result}

    @property
    def model_step_tasks(self) -> dict[str, dict[int, List[str]]]:
        """Get the model step tasks - returns a mapping of model -> step -> list of tasks"""
        result = {}
        for model, task_dict in self.model_task_step_results.items():
            result[model] = defaultdict(list)
            for task, step_dict in task_dict.items():
                for step in step_dict.keys():
                    result[model][step].append(task)
        # Convert defaultdicts to regular dicts and sort task lists
        for model in result:
            result[model] = {step: sorted(tasks) for step, tasks in result[model].items()}
        return result

    @property
    def model_task_steps(self) -> dict[str, dict[str, List[int]]]:
        """Get the model task steps - returns a mapping of model -> task -> list of steps"""
        result = {}
        for model, task_dict in self.model_task_step_results.items():
            result[model] = defaultdict(list)
            for task, step_dict in task_dict.items():
                for step in step_dict.keys():
                    result[model][task].append(step)
        return result
    
    def get_subtasks(self) -> List[str]:
        """Extract all subtasks from stored results"""
        subtasks = set()
        for metadata_dict in self._result_metadata.values():
            for task_metadata in metadata_dict.values():
                for meta in task_metadata.values():
                    result = meta['result']
                    for _, subs in result.group_subtasks.items():
                        subtasks.update(subs)
        return sorted(subtasks)
    
    def __post_init__(self):
        """Initialize the metadata tracking dictionary"""
        if self._result_metadata is None:
            self._result_metadata = defaultdict(lambda: defaultdict(dict))
    
    def _is_baseline(self, model: str) -> bool:
        """Check if a model is the baseline"""
        return self.baseline and self.baseline.lower() in model.lower()
    
    def _should_add_model_step(self, result: EvalResult) -> bool:
        """Check if this model/step should be added to the summary"""
        # Always add baseline
        if self._is_baseline(result.model):
            return True
        
        # Add if model not in summary
        if result.model not in self.models:
            return True
        
        # Add if model in summary but step not in model's steps
        if result.step not in self.model_steps.get(result.model, []):
            return True
        
        return False
    
    def _should_add_task_result(self, result: EvalResult, task_result: TaskResult, step: int) -> bool:
        """Check if this task result should replace existing one"""
        task_name = task_result.task
        model = result.model
        
        # Add if task not in summary
        if task_name not in self.tasks:
            return True
        
        # If we don't have any result for this model/task/step yet, add it
        if model not in self.model_task_step_results:
            return True
        if task_name not in self.model_task_step_results[model]:
            return True
        if step not in self.model_task_step_results[model][task_name]:
            return True
        
        # Check metadata to determine if we should replace
        existing_metadata = self._result_metadata.get(model, {}).get(task_name, {}).get(step)
        if not existing_metadata:
            return True
        
        existing_task = self.model_task_step_results[model][task_name][step]
        
        # Check if new result has metrics that existing doesn't have
        existing_metric_keys = {(m.metric, m.filter) for m in existing_task.metrics}
        new_metric_keys = {(m.metric, m.filter) for m in task_result.metrics}
        if not new_metric_keys.issubset(existing_metric_keys):
            return True
        
        # Replace if limit is higher
        if result.limit > existing_metadata['limit']:
            return True
        
        # Replace if limit is same but result is newer
        if result.limit == existing_metadata['limit'] and result.end_time > existing_metadata['end_time']:
            return True
        
        return False
    
    def add_result(self, result: EvalResult):
        """ Add a result to the summary
            if one of the following is true, add the result to the summary:
                - the model in the result is the baseline
                - the model in the result is not in the summary
                - the model is in the summary but the step is not in the model's steps
            if one of the following is true, add the corresponding task result to the summary's appropriate place:
                - the task in the result is not in the summary
                - the task is in the summary but the metric is not in the task's metrics
                - the task is in the summary and the metric is in the task's metrics but the limit is higher than the limit in the summary
                - the task is in the summary and the metric is in the task's metrics but the limit is the same as the limit in the summary but the result is newer than the result in the summary
        """
        # Handle baseline specially
        if self._is_baseline(result.model):
            for task_result in result.tasks:
                task_name = task_result.task
                # Add baseline task if not exists or should be updated
                if task_name not in self.baseline_task_results or self._should_add_task_result(result, task_result, result.step):
                    self.baseline_task_results[task_name] = task_result
                    # Update metadata for baseline
                    self._result_metadata['__baseline__'][task_name][0] = {
                        'limit': result.limit,
                        'end_time': result.end_time,
                        'result': result
                    }
                    if task_name not in self.tasks:
                        self.tasks.append(task_name)
            return
        
        # Check if we should add this model/step
        if not self._should_add_model_step(result):
            # Even if we don't add the model/step, we might still update individual tasks
            pass
        else:
            # Add model if not exists
            if result.model not in self.models:
                self.models.append(result.model)
                self.model_steps[result.model] = []
                self.model_task_step_results[result.model] = defaultdict(dict)
            
            # Add step if not exists
            if result.step not in self.model_steps[result.model]:
                self.model_steps[result.model].append(result.step)
                self.model_steps[result.model].sort()
        
        # Process each task result
        for task_result in result.tasks:
            task_name = task_result.task
            
            # Check if we should add this task result
            if self._should_add_task_result(result, task_result, result.step):
                # Ensure model exists in results dict
                if result.model not in self.model_task_step_results:
                    self.model_task_step_results[result.model] = defaultdict(dict)
                
                # Add or update the task result
                self.model_task_step_results[result.model][task_name][result.step] = task_result
                
                # Update metadata
                self._result_metadata[result.model][task_name][result.step] = {
                    'limit': result.limit,
                    'end_time': result.end_time,
                    'result': result
                }
                
                # Add task to tasks list if not exists
                if task_name not in self.tasks:
                    self.tasks.append(task_name)
    
    def get_results(self) -> List[EvalResult]:
        """Get all unique EvalResults stored in this summary"""
        results_set = set()
        results_list = []
        
        # Collect from baseline
        for metadata_dict in self._result_metadata.get('__baseline__', {}).values():
            for meta in metadata_dict.values():
                result = meta['result']
                result_id = id(result)
                if result_id not in results_set:
                    results_set.add(result_id)
                    results_list.append(result)
        
        # Collect from regular models
        for model in self.models:
            for task_dict in self._result_metadata.get(model, {}).values():
                for meta in task_dict.values():
                    result = meta['result']
                    result_id = id(result)
                    if result_id not in results_set:
                        results_set.add(result_id)
                        results_list.append(result)
        
        return results_list
    
    @classmethod
    def from_results(cls, results: List[EvalResult], baseline: str = None) -> "EvalSummary":
        """Create an EvalSummary from a list of EvalResults"""
        summary = cls(
            models=[],
            model_steps={},
            tasks=[],
            model_task_step_results=defaultdict(lambda: defaultdict(dict)),
            baseline=baseline or "",
            baseline_task_results={}
        )
        
        for result in results:
            summary.add_result(result)
        
        return summary


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


def plot_results(summary: EvalSummary, output_path: str, include_subtasks: bool):
    """Plot evaluation results from summary"""
    subtasks = summary.get_subtasks()
    print(f"Subtasks: {subtasks}")
    
    metrics_data = defaultdict(lambda: defaultdict(lambda: {'steps': [], 'values': [], 'stderrs': []}))
    baseline_metrics = {}
    
    for task_name, task_result in summary.baseline_task_results.items():
        if not include_subtasks and task_name in subtasks:
            continue
        for metric in task_result.metrics:
            key = f"{task_name}\nmetric:{metric.metric},filter:{metric.filter}"
            baseline_metrics[key] = metric.value
    
    for model in summary.models:
        for task_name in summary.tasks:
            if not include_subtasks and task_name in subtasks:
                continue
            
            task_steps = summary.model_task_step_results.get(model, {}).get(task_name, {})
            for step, task_result in sorted(task_steps.items()):
                for metric in task_result.metrics:
                    key = f"{task_name}\nmetric:{metric.metric},filter:{metric.filter}"
                    metrics_data[key][model]['steps'].append(step)
                    metrics_data[key][model]['values'].append(metric.value)
                    metrics_data[key][model]['stderrs'].append(metric.stderr)
    
    if summary.baseline:
        print(f"Using baseline: {summary.baseline} with {len(baseline_metrics)} metrics")
    
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
            ax.axhline(y=baseline_value, color='#FF6B9D', linestyle='--', linewidth=2, label=f'Baseline: {summary.baseline}', alpha=0.8)
        
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
    parser.add_argument("-x", "--exclude_models", type=str, nargs="*", help="Model regex to exclude from plotting", default=["/storage/hf_models/powen3-8b-500"], required=False)
    parser.add_argument("-b", "--baseline", type=str, help="Baseline model to plot", default='qwen/qwen3-8b-base', required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    results = load_results(eval_dir)
    results = exclude_models(results, args.exclude_models)
    summary = EvalSummary.from_results(results, args.baseline.lower() if args.baseline else None)
    plot_results(summary, args.output_path, args.include_subtasks)

if __name__ == "__main__":
    main()
    