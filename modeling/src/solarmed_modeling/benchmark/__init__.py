import itertools
from pathlib import Path
import datetime
import hjson
from typing import Optional
from loguru import logger
from dataclasses import dataclass, asdict
import pandas as pd

logger.disable("phd_visualizations.utils")

default_sample_rates: list[int] = [5, 30, 60, 300, 500]

@dataclass
class VisualizeValidationConfig:
    subsystem_id: str
    plot_config: dict
    params_str: Optional[str] = None
    out_var_ids: Optional[list[str]] = None
    out_var_units: Optional[list[str]] = None
    pop_var_ids: Optional[list[str]] = None

def benchmark_model(
    model_params: dict[str, float],
    evaluate_model_fn: callable,
    alternatives_to_eval: list[str],
    fixed_model_params: dict[str, float] = None,
    fsm_params: dict[str, float] = None,
    test_ids: list[str] = None, 
    data_path: Path = Path("../../data"), 
    datasets_path: Path = None,
    filenames_data: list[str] = None,
    sample_rates: list[int] = default_sample_rates,
    horizon_times: list[int] | None = None,
    default_files_suffix: str | list[str] = "_solarMED",
    filter_non_active: bool = False, 
    filter_str: str = "",
    save_results: bool = False,
    viz_val_config: Optional[VisualizeValidationConfig] = None,
    output_path: Path = Path("../../results/models_validation"),
    skip_sample_rates_viz: Optional[list[int]] = None,
    skip_horizon_times_viz: Optional[list[int]] = None,
) -> list[dict[str, str | dict[str, float]]]:
    """Benchmark a model by evaluating it on different datasets and sample rates.
    
    This function can be called directly, but initially it was designed to be called from a specific model module,
    which are light wrappers around this function where model-specific parameters are validated and default values are set.

    Args:
        model_params (dict[str, float]): Parameters for the model.
        evaluate_model_fn (callable): Function to evaluate the model.
        alternatives_to_eval (list[str]): List of alternatives to evaluate.
        test_ids (list[str], optional): Identifiers for the tests. Defaults to None.
        data_path (Path, optional): Path to the data directory. Defaults to Path("../../data").
        datasets_path (Path, optional): Path to the datasets directory. Defaults to None.
        filenames_data (list[str], optional): List of filenames for the data. Defaults to None.
        sample_rates (list[int], optional): List of sample rates in seconds. Defaults to [5, 30, 60, 300, 600, 1000].
        default_files_suffix (str, optional): Suffix for the default files. Defaults to "_solarMED".

    Returns:
        list[dict[str, str | dict[str, float]]]: List of dictionaries containing the benchmark results.
    """
    
    from solarmed_modeling.utils import data_preprocessing, data_conditioning # To avoid circular import errors
    
    if not isinstance(default_files_suffix, list):
        default_files_suffix = [default_files_suffix]
    
    sample_rates = default_sample_rates if sample_rates is None else sample_rates
    
    with open(data_path / "variables_config.hjson") as f:
        vars_config = hjson.load(f)
        
    if datasets_path is None:
        datasets_path = data_path / "datasets"

    if filenames_data is None and test_ids is None:
        # Get all files in data_path that end with _solarMED.csv
        filenames_data = [f.name for f in datasets_path.glob(f"*{default_files_suffix[0]}.csv")]
        test_ids = [f.split(f"{default_files_suffix[0]}.csv")[0] for f in filenames_data]

    if filenames_data is None:
        # Validate test_ids are in YYYYMMDD format
        assert all([
            len(test_id) == 8 and 
            int(test_id[:4]) <= datetime.datetime.now().year and 
            1 <= int(test_id[4:6]) <= 12 and 
            1 <= int(test_id[6:8]) <= 31 
            for test_id in test_ids
        ]), "test_ids must be in YYYYMMDD format"

        filenames_data = [f"{test_id}{default_files_suffix[0]}.csv" for test_id in test_ids]
        
    if horizon_times is None:
        horizon_times = [None]

    stats = []
    for idx, test_id in enumerate(test_ids):
        print(f"Processing test {test_id} ({idx+1}/{len(test_ids)})")
        logger.info(f"Processing test {test_id} ({idx+1}/{len(test_ids)})")
        
        # Load data and preprocess data
        try:
            df = data_preprocessing(
                [
                    datasets_path / f"{filenames_data[idx].replace(default_files_suffix[0], suffix)}" 
                    for suffix in default_files_suffix
                ],
                vars_config,
                sample_rate_key=f"{sample_rates[0]}s",
            )
        except FileNotFoundError as e:
            logger.error(f"File not found for test_id {test_id}: {e}")
            continue
        # Condition data
        df = data_conditioning(df, sample_rate_numeric=sample_rates[0], vars_config=vars_config)
        df0 = df.copy()
        
        # Resample data to each sample rate
        dfs = [df.copy().resample(f"{ts}s").mean() for ts in sample_rates] 
        
        # Filter data if required
        if filter_non_active:
            assert len(filter_str) > 0, "filter_str must be provided if filter_non_active is True"
            try:
                for i, (df, sample_rate) in enumerate(zip(dfs, sample_rates)):
                    len0 = len(df)
                    dfs[i] = df[eval(filter_str)]  # modifies element in dfs
                    if len(df) < 3:
                        logger.error(f"Filtered data is empty for test_id {test_id} with filter: {filter_str}")
                        continue
                
                    logger.info(f"[Ts={sample_rate}s] Filtered data went from {len0} to {len(dfs[i])} rows using filter: {filter_str}")
                
                # freq = pd.infer_freq(df.index)
                # if freq is None:
                #     logger.warning("Filtered data has an irregular index. Please provide a filter that does not produce gaps in the data, since it will produce NaNs when resampling to lower sample rates.")
            except Exception as e:
                logger.warning(f"Error evaluating filter_str: {filter_str}. Error: {e}")
                continue

        dfs_mod: list[pd.DataFrame] = []
        try:
            for df_ in dfs:
                for ts, ht in itertools.product(sample_rates, horizon_times):
                    out = evaluate_model_fn(
                        df_, 
                        sample_rate=ts,
                        horizon_time=ht,
                        model_params=model_params, 
                        fixed_model_params=fixed_model_params,
                        fsm_params=fsm_params,
                        alternatives_to_eval=alternatives_to_eval, 
                        base_df=dfs[0],
                    )
                    
                    stats.extend(out[1])
                    dfs_mod.extend(out[0])
                    del out
                            
            # Match sample rates so they can be plot together
            # dfs_mod = [df_.reindex(df.index, method='ffill') for df_ in dfs_mod]
            logger.info(f"Performance metrics are calculated by resampling results to the lowest sample rate ({sample_rates[0]}s)")
            
            # Visualize results figure
            if save_results and viz_val_config is not None:
                from phd_visualizations import save_figure
                from phd_visualizations.test_timeseries import experimental_results_plot
                from phd_visualizations.regression import regression_plot
                
                date_str = df0.index[0].strftime('%Y%m%d')
                
                # Match sample rates so they can be plot together
                # Resample to the base dataframe **AFTER** applying the active filter
                dfs_mod = [df_.reindex(dfs[0].index, method='ffill') for df_ in dfs_mod]
                
                # Save results to csv
                dfs[0].to_csv(output_path / f"out_exp_{viz_val_config.subsystem_id}_{date_str}.csv")
                [
                    df_.to_csv(output_path / f"out_mod_{viz_val_config.subsystem_id}_Ts_{ts}s_H_{ht}s_{date_str}.csv")
                    for (ts, ht), df_ in zip(itertools.product(sample_rates, horizon_times), dfs_mod)
                ]
                
                # Remove variables not wanted in the plots
                if viz_val_config.pop_var_ids is not None:
                    [
                        df_.drop(
                            columns=[col for col in viz_val_config.pop_var_ids if col in df_.columns], 
                            inplace=True
                        ) 
                        for df_ in dfs_mod
                    ]
                
                if viz_val_config.params_str is None:
                    viz_val_config.params_str = ", ".join([f"{k}: {v}" for k, v in asdict(model_params).items()])

                if horizon_times is not None:
                    plt_title_hor_str = f"| Horizon time (H)={horizon_times} s</span>"
                else:
                    plt_title_hor_str = ''

                fig = experimental_results_plot(
                    viz_val_config.plot_config,
                    df0,
                    df_comp=[
                        df_ for (ts,ht), df_ in zip(itertools.product(sample_rates, horizon_times), dfs_mod) 
                        if (ts not in (skip_sample_rates_viz or [])) and (ht not in (skip_horizon_times_viz or []))
                    ],
                    comp_trace_labels=[
                        "[" + 
                        ", ".join(
                            part for part in [
                                f"Ts={ts}s" if len(sample_rates) > 1 else None,
                                f"H={ht/3600:0.1f}h" if (horizon_times is not None and len(horizon_times) > 1) else None,
                            ] if part is not None
                        ) + "]"
                        for ts, ht in itertools.product(sample_rates, horizon_times)
                        if (ts not in (skip_sample_rates_viz or [])) and (ht not in (skip_horizon_times_viz or []))
                    ] * len(alternatives_to_eval),
                    # {df.index[0].strftime('%d/%m/%Y')}
                    # ɣ: {model_params.gamma:.4f}
                    title_text=f"<b>{viz_val_config.subsystem_id.capitalize().replace('_', ' ')}</b> model validation<br><span style='font-size: 13px;'>{viz_val_config.params_str} | T<sub>s</sub>={sample_rates}s {plt_title_hor_str}</span>",
                    vars_config=vars_config,
                    resample=False,
                    template="plotly_white",
                )
                # Save figure
                save_figure(
                    figure_name=f"{viz_val_config.subsystem_id}_validation_{date_str}",
                    figure_path=output_path,
                    fig=fig, formats=('png', 'html'), 
                    width=fig.layout.width, height=fig.layout.height, scale=2
                )
                
                if viz_val_config.out_var_ids is not None:
                    for df_mod, (ts, ht) in zip(dfs_mod, itertools.product(sample_rates, horizon_times)):
                        fig = regression_plot(
                            df_ref=dfs[0],
                            df_mod=df_mod,
                            var_ids=viz_val_config.out_var_ids,
                            units=viz_val_config.out_var_units,
                            show_error_metrics=["r2", "mae", "mape"],
                            inline_error_metrics_text=True,
                            legend_pos="side",   
                        )
                        save_figure(
                            figure_name=f"{viz_val_config.subsystem_id}_regression_Ts_{ts}s_H_{ht}s_{date_str}",
                            figure_path=output_path,
                            fig=fig, formats=('png', 'html'), 
                            width=fig.layout.width, height=fig.layout.height, scale=2
                        )
                
        except (KeyError, AssertionError) as e:
            print(f"Failed to evaluate {test_id}: {e}")
            logger.error(f"Failed to evaluate {test_id}: {e}")
                    
    return stats