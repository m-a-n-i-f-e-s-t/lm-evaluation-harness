import argparse
import json
import re
from pathlib import Path
from typing import List, Any, Annotated

from matplotlib import pyplot as plt
from pydantic import BaseModel, model_validator, Field
import polars as pl


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
    filename: str = Field(default="")
    param: float = Field(default=0)
    flavor: str = Field(default="")
    
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
        self.model = self.model.lower().rstrip('/')
        if self.model.endswith('/phase1') or self.model.endswith('/phase2') or self.model.endswith('/phase3'):
            self.model = '/'.join(self.model.split('/')[:-1])
        param_match = re.search(r"(\d+(?:\.\d+)?)b", self.model)
        if param_match:
            self.param = float(param_match.group(1))
        if 'qwen3' in self.model:
            if 'base' in self.model:
                self.flavor = 'qwen3-base'
            else:
                self.flavor = 'qwen3'
        elif 'nemo' in self.model:
            if 'base' in self.model:
                self.flavor = 'nemo-base'
            else:
                self.flavor = 'nemo'
        self.results = {}
        return self


def load_results(eval_dir: Path) -> List[EvalResult]:
    results = []
    for result_file in eval_dir.glob("*/**/results_*.json"):
        with open(result_file, "r") as f:
            r = json.load(f)
            result = EvalResult(**r, filename=result_file.name)
            results.append(result)
    return results


def exclude_models(results: List[EvalResult], exclude_models: List[str]) -> List[EvalResult]:
    """Exclude models from the results"""
    excluded_models = [r.model for r in results if any(exclude_model in r.model for exclude_model in exclude_models)]
    print(f"Excluded models: {excluded_models}")
    return [r for r in results if r.model not in excluded_models]


def flatten_results_to_dataframe(results: List[EvalResult]) -> pl.DataFrame:
    """Flatten all results into a Polars DataFrame
    
    Returns a DataFrame with columns:
    - model: str
    - step: int
    - param: float
    - flavor: str
    - task: str
    - alias: str
    - metric: str
    - filter: str
    - value: float
    - stderr: float
    - limit: int
    - end_time: float
    - batch_size: int
    - filename: str
    """
    rows = []
    
    for result in results:
        for task_result in result.tasks:
            for metric in task_result.metrics:
                rows.append({
                    'model': result.model,
                    'step': result.step,
                    'param': result.param,
                    'flavor': result.flavor,
                    'task': task_result.task,
                    'alias': task_result.alias,
                    'metric': metric.metric,
                    'filter': metric.filter,
                    'value': metric.value,
                    'stderr': metric.stderr,
                    'limit': result.limit,
                    'end_time': result.end_time,
                    'batch_size': result.batch_size,
                    'filename': result.filename,
                })
    
    return pl.DataFrame(rows)


def get_subtasks_from_results(results: List[EvalResult]) -> List[str]:
    """Extract all subtasks from results"""
    subtasks = set()
    for result in results:
        for task_group, subs in result.group_subtasks.items():
            subtasks.update(subs)
    return sorted(subtasks)


def deduplicate_results(df: pl.DataFrame, baseline: str = None) -> pl.DataFrame:
    """Deduplicate results using the following rules:
    
    For each combination of (model, task, step, metric, filter):
    1. Keep the one with highest limit
    2. If limit is the same, keep the one with latest end_time
    
    Args:
        df: DataFrame with flattened results
        baseline: Optional baseline model name to mark
    
    Returns:
        Deduplicated DataFrame with additional 'is_baseline' column
    """
    # Add is_baseline column
    if baseline:
        df = df.with_columns(
            pl.col('model').str.contains(baseline.lower()).alias('is_baseline')
        )
    else:
        df = df.with_columns(
            pl.lit(False).alias('is_baseline')
        )
    
    # Sort by limit (descending) and end_time (descending)
    # Then group by (model, task, step, metric, filter) and take first
    df_deduped = (
        df.sort(['limit', 'end_time'], descending=True)
        .group_by(['model', 'task', 'step', 'metric', 'filter'])
        .first()
    )
    
    return df_deduped


def plot_results_df(df: pl.DataFrame, output_path: str, include_subtasks: bool, subtasks: List[str]):
    """Plot evaluation results from DataFrame
    
    Args:
        df: DataFrame with deduplicated results
        output_path: Path to save the plot
        include_subtasks: Whether to include subtasks in the plot
        subtasks: List of subtask names
    """
    # Filter out subtasks if needed
    if not include_subtasks:
        df = df.filter(~pl.col('task').is_in(subtasks))
    
    # Separate baseline and non-baseline results
    baseline_df = df.filter(pl.col('is_baseline'))
    non_baseline_df = df.filter(~pl.col('is_baseline'))
    
    # Get unique models (excluding baseline)
    models = sorted(non_baseline_df['model'].unique().to_list())
    print(f"Models: {models}")
    
    # Get all unique metric keys (task + metric + filter combinations)
    metric_keys = (
        df.select([
            pl.concat_str([
                pl.col('task'),
                pl.lit('\nmetric:'),
                pl.col('metric'),
                pl.lit(',filter:'),
                pl.col('filter')
            ]).alias('metric_key')
        ])
        .unique()
        .sort('metric_key')
        ['metric_key'].to_list()
    )
    
    n_metrics = len(metric_keys)
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
    
    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        
        # Parse metric_key back to task, metric, filter
        parts = metric_key.split('\nmetric:')
        task = parts[0]
        metric_filter = parts[1]
        metric_name, filter_name = metric_filter.split(',filter:')
        
        # Plot baseline if exists
        baseline_data = baseline_df.filter(
            (pl.col('task') == task) & 
            (pl.col('metric') == metric_name) & 
            (pl.col('filter') == filter_name)
        )
        
        if len(baseline_data) > 0:
            baseline_value = baseline_data['value'][0]
            baseline_model = baseline_data['model'][0]
            ax.axhline(y=baseline_value, color='#FF6B9D', linestyle='--', 
                      linewidth=2, label=f'Baseline: {baseline_model}', alpha=0.8)
        
        # Plot each model
        for model in models:
            model_data = (
                non_baseline_df.filter(
                    (pl.col('model') == model) &
                    (pl.col('task') == task) & 
                    (pl.col('metric') == metric_name) & 
                    (pl.col('filter') == filter_name)
                )
                .sort('step')
            )
            
            if len(model_data) == 0:
                continue
            
            steps = model_data['step'].to_list()
            values = model_data['value'].to_list()
            stderrs = model_data['stderr'].to_list()
            
            line = ax.plot(steps, values, marker='o', label=model, linewidth=2, markersize=4)
            color = line[0].get_color()
            
            lower = [v - s for v, s in zip(values, stderrs)]
            upper = [v + s for v, s in zip(values, stderrs)]
            ax.fill_between(steps, lower, upper, alpha=0.2, color=color)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(metric_key, fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_param_scaling_df(df: pl.DataFrame, output_path: str, include_subtasks: bool, subtasks: List[str]):
    """Plot parameter scaling as scatter plot from DataFrame
    
    Args:
        df: DataFrame with deduplicated results
        output_path: Path to save the plot
        include_subtasks: Whether to include subtasks in the plot
        subtasks: List of subtask names
    """
    # Filter out subtasks if needed
    if not include_subtasks:
        df = df.filter(~pl.col('task').is_in(subtasks))
    
    # Filter out models without param info or flavor
    df = df.filter((pl.col('param') > 0) & (pl.col('flavor') != ''))
    
    # Separate baseline and non-baseline results
    baseline_df = df.filter(pl.col('is_baseline'))
    non_baseline_df = df.filter(~pl.col('is_baseline'))
    
    # For param scaling, use the last step for each model
    # First sort by step, then group and take last
    non_baseline_df = (
        non_baseline_df
        .sort('step')
        .group_by(['model', 'task', 'metric', 'filter', 'param', 'flavor'])
        .agg([
            pl.col('step').last().alias('step'),
            pl.col('value').last().alias('value'),
            pl.col('stderr').last().alias('stderr'),
            pl.col('is_baseline').last().alias('is_baseline'),
            pl.col('alias').last().alias('alias')
        ])
    )
    
    # Get unique flavors
    flavors = sorted(non_baseline_df['flavor'].unique().to_list())
    print(f"Flavors: {flavors}")
    
    # Get all unique metric keys
    metric_keys = (
        df.select([
            pl.concat_str([
                pl.col('task'),
                pl.lit('\nmetric:'),
                pl.col('metric'),
                pl.lit(',filter:'),
                pl.col('filter')
            ]).alias('metric_key')
        ])
        .unique()
        .sort('metric_key')
        ['metric_key'].to_list()
    )
    
    n_metrics = len(metric_keys)
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
    
    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        
        # Parse metric_key back to task, metric, filter
        parts = metric_key.split('\nmetric:')
        task = parts[0]
        metric_filter = parts[1]
        metric_name, filter_name = metric_filter.split(',filter:')
        
        # Plot baseline if exists
        baseline_data = baseline_df.filter(
            (pl.col('task') == task) & 
            (pl.col('metric') == metric_name) & 
            (pl.col('filter') == filter_name)
        )
        
        if len(baseline_data) > 0:
            baseline_value = baseline_data['value'][0]
            baseline_model = baseline_data['model'][0]
            ax.axhline(y=baseline_value, color='#FF6B9D', linestyle='--', 
                      linewidth=2, label=f'Baseline: {baseline_model}', alpha=0.8)
        
        # Plot each flavor
        for flavor in flavors:
            flavor_data = (
                non_baseline_df.filter(
                    (pl.col('flavor') == flavor) &
                    (pl.col('task') == task) & 
                    (pl.col('metric') == metric_name) & 
                    (pl.col('filter') == filter_name)
                )
                .sort('param')
            )
            
            if len(flavor_data) == 0:
                continue
            
            params = flavor_data['param'].to_list()
            values = flavor_data['value'].to_list()
            stderrs = flavor_data['stderr'].to_list()
            
            # Plot line with markers and error bands
            line = ax.plot(params, values, marker='o', label=flavor, linewidth=2, markersize=6)
            color = line[0].get_color()
            
            # Add error bands
            lower = [v - s for v, s in zip(values, stderrs)]
            upper = [v + s for v, s in zip(values, stderrs)]
            ax.fill_between(params, lower, upper, alpha=0.2, color=color)
        
        ax.set_xlabel('Parameters (Billions)')
        ax.set_ylabel('Performance')
        ax.set_title(metric_key, fontsize=14)
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
    parser.add_argument("-x", "--exclude_models", type=str, nargs="*", help="Model regex to exclude from plotting", default=["/storage/hf_models/powen3-8b-500", "powen3-8b-3000"], required=False)
    parser.add_argument("-b", "--baseline", type=str, help="Baseline model to plot", default='qwen/qwen3-8b-base', required=False)
    parser.add_argument("--param_scaling", action="store_true", help="Plot parameter scaling instead of step", default=False, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    
    # Load and process results using DataFrame approach
    results = load_results(eval_dir)
    results = exclude_models(results, args.exclude_models)
    
    # Flatten to DataFrame
    df = flatten_results_to_dataframe(results)
    print(f"Loaded {len(df)} metric records from {len(results)} result files")
    
    # Deduplicate
    df = deduplicate_results(df, args.baseline.lower() if args.baseline else None)
    print(f"After deduplication: {len(df)} metric records")
    
    # Get subtasks
    subtasks = get_subtasks_from_results(results)
    print(f"Subtasks: {subtasks}")
    
    # Plot using DataFrame-based functions
    if args.param_scaling:
        plot_param_scaling_df(df, args.output_path, args.include_subtasks, subtasks)
    else:
        plot_results_df(df, args.output_path, args.include_subtasks, subtasks)

if __name__ == "__main__":
    main()
    