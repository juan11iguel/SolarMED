import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

def visualize_benchmark(results: list[dict[str, str | dict[str, float]]], output_unit: str = "", title: str = "Benchmark results", y_vars_config: list[dict[str, str]] = None, width: int = None) -> list[go.Figure]:

    """Visualize benchmark results
    
    Args:
        results (list[dict]): List of dictionaries with the results of a model evaluation
        output_unit (str, optional): Unit of the output variable. Defaults to "".
        title (str, optional): Title of the plot. Defaults to "Benchmark results".
        y_vars_config (list[dict[str, str]], optional): List of dictionaries with the configuration of the y variables to plot. Defaults to None.
        width (int, optional): Width of the plot. Defaults to

    Returns:
        list[go.Figure]: List of plotly figures
    """

    # To modify the gaps between subplots, you can modify the domain of their xaxis
    def domains_calculator(gap, n_plot):
        """
        little helper to calculate the range of subplots domains, by DIDIER Sébastien
        https://community.plotly.com/t/control-distance-between-stacked-bars/75303/4
        """
        plot_width = (1 - (n_plot-1) * gap) / n_plot
        return [
            [i * (plot_width + gap), i * (plot_width + gap) + plot_width]
            for i in range(n_plot)
        ]
    
    # Handle nested attributes in y_var
    def get_nested_attr(d, attr):
        keys = attr.split('.')
        for key in keys:
            d = d.get(key, None)
        return d

    def generate_plot(y_var: str, y_label: str) -> go.Figure | None:
        """ Generate a plot for a given y_var and y_label
        
        Based on DIDIER Sébastien suggestion in [plotly forum](https://community.plotly.com/t/control-distance-between-stacked-bars/75303/4)

        Args:
            y_var (str): Name of the variable in `results`
            y_label (str): Label for the y axis

        Returns:
            go.Figure | None: Plotly figure object
        """
        
        if get_nested_attr(results[0], y_var) is None:
            logger.warning(f"Attribute {y_var} not found in results, skipping plot")
            
            return None
        
        data["y"] = [get_nested_attr(d, y_var) for d in results]

        fig = px.bar(data, x="x", y="y", color="color", facet_col="facet_col", pattern_shape="color")

        # little customisations
        fig.for_each_annotation(lambda a: a.update(text=''))# remove the facet titles
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            barmode='group', 
            legend=dict(title='Color legend', x=0.8, y=1.3),
            # margin=dict(l=0, r=0, t=100, b=20),
            width = n_plots * 300 if width is None else width,
        )
        fig.update_yaxes(title_text="", ticks="outside", linecolor="black", row=1, col=1)

        domains = domains_calculator(gap=0.05, n_plot=n_plots)

        [fig.update_xaxes(title_text=f"<b>{test_id}</b>", domain=domains[idx], row=1, col=idx+1) for idx, test_id in enumerate(unique_test_ids)]
        fig.update_yaxes(title_text=y_label, row=1, col=1)

        # and to modify the gap between the subplots bars, you can modify their width
        fig.update_traces(width=0.3)
        
        return fig
    
    unique_test_ids = list(set([d['test_id'] for d in  results]))
    n_plots = len(unique_test_ids)
    
    # Prepare data
    data = {
        'x': [f'Ts={d["sample_rate"]} s' for d in results],
        'color': [d['alternative'] for d in results],
        'facet_col': [d['test_id'] for d in results],
    }
    
    if y_vars_config is not None:
        assert all('y_var' in y_var_config and 'y_label' in y_var_config for y_var_config in y_vars_config), "y_vars_config must contain 'y_var' and 'y_label' keys for each element" 
    else:
        y_vars_config = [
            {
                "y_var": "elapsed_time",
                "y_label": "Elapsed time [s]",
            },
            {
                "y_var": "metrics.MAE",
                "y_label": f"Mean Absolute Error [{output_unit}]",
            },
            {
                "y_var": "metrics.MSE",
                "y_label": f"Mean Squared Error [{output_unit}²]",
            },
            {
                "y_var": "metrics.RMSE",
                "y_label": f"Root Mean Squared Error [{output_unit}²]",
            },
            {
                "y_var": "metrics.R2",
                "y_label": "R²",
            },
        ]
    
    figs = [generate_plot(y_var=y_var_config["y_var"], y_label=y_var_config["y_label"]) for y_var_config in y_vars_config if get_nested_attr(results[0], y_var_config["y_var"]) is not None]
    # Add title to the first subplot
    figs[0].update_layout(
        title=title
    )
    
    return figs