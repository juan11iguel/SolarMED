from pathlib import Path
from typing import Optional
import hjson
from . import FixedModelParameters, supported_eval_alternatives
from .utils import evaluate_model
from solarmed_modeling import benchmark

model_id: str = 'med'

def benchmark_model(
    model_params: None = None, 
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[str] = ["standard"],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    datasets_path: Path = None,
    output_path: Path = Path("../../../results/models_validation"),
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 400, 1000],
    default_files_suffix: str = "_MED",
    filter_non_active: bool = True,
    filter_str: str = '(df["qmed_d"] > 0) & (df["Tmed_c_out"] - df["Tmed_c_in"] > 2) & (df["Tmed_s_in"] - df["Tmed_s_out"] > 2)',
    save_results: bool = False,
    viz_val_config: Optional[benchmark.VisualizeValidationConfig] = None,
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    if save_results and viz_val_config is None:
        viz_val_config = benchmark.VisualizeValidationConfig(
            subsystem_id = model_id,
            plot_config = hjson.loads(Path('/workspaces/SolarMED/modeling/data/plt_config_med.json').read_text()),
            params_str = "",
            out_var_ids = ["qmed_d", "qmed_c", "Tmed_s_out"],
            out_var_units = ['m³/h', 'm³/h', '°C']
        )
        for plot in viz_val_config.plot_config["plots"].values():
            if "show_active" in list(plot.keys()):
                plot["show_active"] = False
                plot["tigth_vertical_spacing"] = False
        viz_val_config.plot_config["plots"]["med_temperatures"]["title"] = "Temperatures"
        viz_val_config.plot_config["plots"]["med_flows"]["title"] = "Flows"
    
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
