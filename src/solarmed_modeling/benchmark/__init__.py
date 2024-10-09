from pathlib import Path
import datetime
import hjson
from loguru import logger

def benchmark_model(
    model_params: dict[str, float],
    evaluate_model_fn: callable,
    alternatives_to_eval: list[str],
    test_ids: list[str] = None, 
    data_path: Path = Path("../../data"), 
    datasets_path: Path = None,
    filenames_data: list[str] = None,
    sample_rates: list[int] = [5, 30, 60, 300, 600, 1000],
    default_files_suffix: str = "_solarMED",
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
    
    with open(data_path / "variables_config.hjson") as f:
        vars_config = hjson.load(f)
        
    if datasets_path is None:
        datasets_path = data_path / "datasets"

    if filenames_data is None and test_ids is None:
        # Get all files in data_path that end with _solarMED.csv
        filenames_data = [f.name for f in datasets_path.glob(f"*{default_files_suffix}.csv")]
        test_ids = [f.split(f"{default_files_suffix}.csv")[0] for f in filenames_data]

    if filenames_data is None:
        # Validate test_ids are in YYYYMMDD format
        assert all([
            len(test_id) == 8 and 
            int(test_id[:4]) <= datetime.datetime.now().year and 
            1 <= int(test_id[4:6]) <= 12 and 
            1 <= int(test_id[6:8]) <= 31 
            for test_id in test_ids
        ]), "test_ids must be in YYYYMMDD format"

        filenames_data = [f"{test_id}{default_files_suffix}.csv" for test_id in test_ids]

    stats = []
    for idx, test_id in enumerate(test_ids):
        logger.info(f"Processing test {test_id} ({idx+1}/{len(test_ids)})")
        
        # Load data and preprocess data
        df = data_preprocessing(
            datasets_path / f"{filenames_data[idx]}",
            vars_config,
            sample_rate_key=f"{sample_rates[0]}s",
        )
        # Condition data
        df = data_conditioning(df, sample_rate_numeric=sample_rates[0])
        
        # Resample data to each sample rate
        dfs = [df.copy().resample(f"{ts}s").mean() for ts in sample_rates] 

        for df_, ts in zip(dfs, sample_rates):
            out = evaluate_model_fn(df_, ts, model_params, alternatives_to_eval=alternatives_to_eval, base_df=dfs[0])
            stats.extend(out[1])
            del out
                        
        # Match sample rates so they can be plot together
        # dfs_mod = [df_.reindex(df.index, method='ffill') for df_ in dfs_mod]
        logger.info(f"Performance metrics are calculated by resampling results to the lowest sample rate ({sample_rates[0]}s)")
    
    return stats