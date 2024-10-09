from pathlib import Path
from . import evaluate_model, ModelParameters, supported_eval_alternatives
from solarmed_modeling.utils import benchmark

model_id: str = 'heat_exchanger'

def benchmark_model(
    model_params: ModelParameters, 
    alternatives_to_eval: list[str] = ["standard", "constant-water-props"],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    datasets_path: Path = None,
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 60, 300, 600, 1000],
    default_files_suffix: str = "_solarMED",
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    return benchmark.benchmark_model(
        model_params=model_params,
        evaluate_model_fn=evaluate_model,
        alternatives_to_eval=alternatives_to_eval,
        test_ids=test_ids,
        data_path=data_path,
        datasets_path=datasets_path,
        filenames_data=filenames_data,
        sample_rates=sample_rates,
        default_files_suffix=default_files_suffix,
    )
