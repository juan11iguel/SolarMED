from pathlib import Path
from typing import Optional
import hjson
from . import ModelParameters, FixedModelParameters, supported_eval_alternatives, FsmParameters
from .utils import evaluate_model
from solarmed_modeling import benchmark

model_id: str = 'solar_med'

def benchmark_model(
    model_params: ModelParameters = ModelParameters(),
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    fsm_params: FsmParameters = FsmParameters(),
    alternatives_to_eval: list[str] = ["constant-water-props"],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../../data"), 
    output_path: Path = Path("../../../results/models_validation"),
    datasets_path: Path = None,
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 400],
    default_files_suffix: list[str] = ["_solarMED", "_MED"],
    filter_non_active: bool = False,
    filter_str: str = '',
    save_results: bool = False,
    viz_val_config: Optional[benchmark.VisualizeValidationConfig] = None,
) -> list[dict[str, str | dict[str, float]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Unsupported evaluation alternatives: {alternatives_to_eval}. Supported alternatives are: {supported_eval_alternatives}"
    
    if save_results and viz_val_config is None:
        viz_val_config = benchmark.VisualizeValidationConfig(
            subsystem_id = model_id,
            plot_config = hjson.loads(Path('/workspaces/SolarMED/modeling/data/plot_config.hjson').read_text()),
            params_str = "",
            out_var_ids = ["Tsf_in", "Tsf_out", "Thx_s_out", 'Tts_h_t', 'Tts_h_m', 'Tts_h_b', 'Tts_c_t', 'Tts_c_m', 'Tts_c_b', "qmed_d", "qmed_c"],
            out_var_units = ['°C']*9 + ['m³/h']*2,
            pop_var_ids = ['I', 'Tamb', 'Tmed_c_in', 'qsf', 'qhx_p', 'qhx_s']
        )
        viz_val_config.plot_config["plots"]["med_temperatures"].pop("traces_right")
        # for plot in viz_val_config.plot_config["plots"].values():
        #     if "show_active" in list(plot.keys()):
        #         plot["show_active"] = False
        #         plot["tigth_vertical_spacing"] = False
        # viz_val_config.plot_config["plots"]["med_temperatures"]["title"] = "Temperatures"
        # viz_val_config.plot_config["plots"]["med_flows"]["title"] = "Flows"
    
    return benchmark.benchmark_model(
        model_params=model_params,
        fixed_model_params=fixed_model_params,
        fsm_params=fsm_params,
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
