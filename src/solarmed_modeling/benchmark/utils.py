from pathlib import Path
import datetime
import json
from loguru import logger

def export_benchmark_results(model_id: str, results: list[dict[str, str | dict[str, float]]], output_path: Path):
    """Export benchmark results to a json file

    Args:
        model_id (str): Identifier for the model
        results (list[dict[str, str  |  dict[str, float]]]): List of dictionaries with th benchmark results from evaluating `benchmark_model`
        output_path (Path): Path to the output file
    """
    
    date_str: str = datetime.datetime.now().strftime("%Y%m%d")
    
    if not output_path.exists():
        data = {}
    else:
        with open(output_path, "r") as f:
            data = json.load(f)
    
        
    if date_str not in data:
        data[date_str] = {model_id: results}
    else:
        if model_id in data[date_str]:
            logger.warning(f"Results for {model_id}-{date_str} already exist in {output_path}. Overwriting...")

        data[date_str][model_id] = results
    
    with open(output_path, "w") as f:
        json.dump(data, f, default=str, indent=4)
        
    logger.info(f"Results for {model_id}-{date_str} exported to {output_path}")
    
    
def import_benchmark_results(results_path: Path, eval_date_str: str = None, model_id: str = None) -> dict:
    """Import benchmark results from a json file

    Args:
        results_path (Path): Path to the output file
        eval_date_str (str, optional): Date string when the benchmark was evaluated
        model_id (str, optional): Identifier for the model

    Returns:
        dict: Dictionary with the results
    """
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    stats = {} 
    
    if eval_date_str is None:
        stats = data
    else:
        if eval_date_str not in data:
            raise ValueError(f"No evaluation results found for provided date {eval_date_str}, available are: {list(data.keys())}")
        stats = data[eval_date_str]
    
    if model_id is not None:
        if model_id not in stats:
            raise ValueError(f"Model {model_id} not found in results for {eval_date_str}, available are: {list(stats.keys())}")
        stats = stats[model_id]
            
    return stats