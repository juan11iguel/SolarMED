from pathlib import Path
from . import ModelParameters, FixedModelParameters, supported_eval_alternatives
from .utils import evaluate_model
from solarmed_modeling import benchmark

model_id: str = 'solar_med'

def benchmark_model(
    model_params: ModelParameters = ModelParameters(),
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[str] = supported_eval_alternatives,
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    datasets_path: Path = None,
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 60, 300, 500, 800],
    default_files_suffix: str = "_solarMED",
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    return benchmark.benchmark_model(
        model_params=model_params,
        fixed_model_params=fixed_model_params,
        evaluate_model_fn=evaluate_model,
        alternatives_to_eval=alternatives_to_eval,
        test_ids=test_ids,
        data_path=data_path,
        datasets_path=datasets_path,
        filenames_data=filenames_data,
        sample_rates=sample_rates,
        default_files_suffix=default_files_suffix,
    )
