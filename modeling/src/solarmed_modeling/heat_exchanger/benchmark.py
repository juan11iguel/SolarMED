from pathlib import Path
from typing import Optional
import hjson
from . import ModelParameters, supported_eval_alternatives
from .utils import evaluate_model
from solarmed_modeling import benchmark

model_id: str = 'heat_exchanger'

def benchmark_model(
    model_params: ModelParameters, 
    fixed_model_params: None = None,
    alternatives_to_eval: list[str] = ["standard", "constant-water-props"],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    datasets_path: Path = None,
    output_path: Path = Path("../../../results/models_validation"),
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 400, 1000],
    default_files_suffix: str = "_solarMED",
    filter_non_active: bool = True,
    filter_str: str = '(df["qhx_p"] > 0) & (df["qhx_s" ] > 0)',
    save_results: bool = False,
    viz_val_config: Optional[benchmark.VisualizeValidationConfig] = None,
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    if save_results and viz_val_config is None:
        viz_val_config = benchmark.VisualizeValidationConfig(
            subsystem_id = model_id,
            plot_config = hjson.loads(Path('/workspaces/SolarMED/modeling/data/plt_config_heat_exchanger.json').read_text()),
            params_str = f"UA: {model_params.UA:.4e} (W/K)",
            out_var_ids = ['Thx_p_out', 'Thx_s_out'],
            out_var_units = ['°C', '°C']
        )
        
        viz_val_config.plot_config["plots"].pop("effectiveness", None) # Remove effectiveness plot since the experimental data does not have it and raises an error
    
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
        filter_non_active=filter_non_active,
        filter_str=filter_str,
        save_results=save_results,
        viz_val_config=viz_val_config,
        output_path=output_path
    )
