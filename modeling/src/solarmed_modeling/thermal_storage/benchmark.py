from pathlib import Path
from typing import Optional
import hjson
from itertools import product

from solarmed_modeling import benchmark
from . import ModelParameters, supported_eval_alternatives, FixedModelParameters
from .utils import evaluate_model

model_id: str = 'thermal_storage'

def benchmark_model(
    model_params: ModelParameters, 
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[str] = ["standard", "constant-water-props"],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    datasets_path: Path = None,
    output_path: Path = Path("../../../results/models_validation"),
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 400, 1000],
    default_files_suffix: str = "_solarMED",
    filter_non_active: bool = False,
    filter_str: str = '',
    save_results: bool = False,
    viz_val_config: Optional[benchmark.VisualizeValidationConfig] = None,
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    if save_results and viz_val_config is None:
        viz_val_config = benchmark.VisualizeValidationConfig(
            subsystem_id = model_id,
            plot_config = hjson.loads(Path('/workspaces/SolarMED/modeling/data/plt_config_thermal_storage.json').read_text()),
            params_str = "",
            out_var_ids = ['Tts_h_t', 'Tts_h_m', 'Tts_h_b', 'Tts_c_t', 'Tts_c_m', 'Tts_c_b'],
            out_var_units = ['°C']*6,
        )
        
        pos_keys = ["top", "med", "bottom"]
        for pos_key, tank_key in product(pos_keys, ["hot", "cold"]):
            plot_id = f"temperatures_{tank_key}_{pos_key}"
            key_idx = pos_keys.index(pos_key)
            
            viz_val_config.plot_config["plots"][plot_id]["title"] = f"H={getattr(model_params, f'UA_{tank_key[0]}')[key_idx]:.2e} W/K, V={getattr(model_params, f'V_{tank_key[0]}')[key_idx]:.2f} m³"
            viz_val_config.plot_config["plots"][plot_id]["ylabels_left"] = [f"{tank_key.capitalize()} tank<br>{pos_key.capitalize()}<br>°C"]
        
    return benchmark.benchmark_model(
        model_params=model_params,
        fixed_model_params=fixed_model_params,
        evaluate_model_fn=evaluate_model,
        alternatives_to_eval=alternatives_to_eval,
        test_ids=test_ids,
        data_path=data_path,
        datasets_path=datasets_path,
        output_path=output_path,
        filenames_data=filenames_data,
        sample_rates=sample_rates,
        default_files_suffix=default_files_suffix,
        filter_non_active=filter_non_active,
        filter_str=filter_str,
        save_results=save_results,
        viz_val_config=viz_val_config,
    )
