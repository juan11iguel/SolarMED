from enum import Enum
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from solarmed_modeling.visualization.fsm import get_coordinates_edge
from solarmed_modeling.fsms.utils import get_solarmed_individual_states
# from solarmed_optimization.utils import get_nested_attr


def highlight_path(fig: go.Figure, paths_df: pd.DataFrame, paths_values_df: pd.DataFrame, 
                   selected_path_idx: int, initial_state: Enum, nodes_df: pd.DataFrame, 
                   base_title: str, shift: int = None, nodes_comp_dfs: list[pd.DataFrame] = None) -> go.Figure:
    if initial_state is None:
        return
    
    if shift is None:
        if nodes_comp_dfs is not None:
            temp = nodes_comp_dfs[0]
            shift = len(temp[ temp["step_idx"] == 0 ]["state"])
            print("Even though the argument `shift` is optional, it's better to provide it in order to avoid re-calculating it in the loop")
            del temp
    
    # Find all the paths that start from the initial state
    # I bet doing this is expensive, but I don't know how to compute it only when changing the initial state
    path_idxs_matching_initial_state: list[int] = paths_df[paths_values_df["0"] == initial_state.value].index.tolist()
    
    if selected_path_idx < len(path_idxs_matching_initial_state):
        path_idx = path_idxs_matching_initial_state[selected_path_idx]  
    else: 
        # Choose the last one
        path_idx = path_idxs_matching_initial_state[-1]
    path = paths_df.iloc[path_idx]
    path_str: list[str] = [state.name for state in paths_df.iloc[path_idx]]
    
    print(f"Actual number of available paths: {len(path_idxs_matching_initial_state)}")
    print(f"Selected path {path_idx}: {path_str}")
    # Somehow build the path coordinates from the list of paths

    x1 = []
    y1 = []
    if nodes_comp_dfs is not None:
        x2 = []
        y2 = []
        nodes_df_ = nodes_comp_dfs
    else:
        x2 = None
        y2 = None
        nodes_df_ = [nodes_df]
        
    for step_idx in range(0, len(path) - 1, 1):
        # If the naming scheme changes for whatever reason, this will break
        if nodes_comp_dfs is not None:
            state_1_val, state_2_val = get_solarmed_individual_states(path[str(step_idx)], return_format="value")
            state_1_next_val, state_2_next_val = get_solarmed_individual_states(path[str(step_idx+1)], return_format="value")
        else:
            state_1_val = path[str(step_idx)].value
            state_1_next_val = path[str(step_idx + 1)].value
            
        # First path
        src_node_id = f'step{step_idx:03d}_{state_1_val}'
        dst_node_id = f'step{step_idx + 1:03d}_{state_1_next_val}'
        x_aux, y_aux = get_coordinates_edge(src_node_id, dst_node_id, nodes_df=nodes_df_[0])

        x1 += x_aux
        y1 += y_aux
        
        # Second path
        if nodes_comp_dfs is not None:
            src_node_id = f'step{step_idx:03d}_{state_2_val}'
            dst_node_id = f'step{step_idx + 1:03d}_{state_2_next_val}'
            x_aux, y_aux = get_coordinates_edge(src_node_id, dst_node_id, nodes_df=nodes_df_[1], y_shift=shift)

            x2 += x_aux
            y2 += y_aux
        
    with fig.batch_update():
        # First deactivate departing highlights
        fig.data[-2].x = x2
        fig.data[-2].y = y2

        # Then use arriving trace container to include the path
        fig.data[-1].x = x1
        fig.data[-1].y = y1
        
        fig.layout.title.text = f'<b>{base_title}</b><br><span style="font-size: 11px;">Selected path {path_idx} out of {len(paths_df)}: {path_str}</span></br>'
        
    return fig


def update_n_paths_plot(fig: go.Figure, data: list[dict], initial_state: str) -> go.Figure:
    # with fig.batch_update():
        # First deactivate departing highlights
    fig.data[0].y = [d['n_paths_per_initial_state'][initial_state] for d in data]

    return fig


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
    
def generate_n_paths_per_initial_state_bar_plot(metadata: dict, system: str, initial_states: list[str], 
                                                y_label: str = "n of paths") -> go.Figure | None:
    """ Generate a plot for a given y_var and y_label
    
    Based on DIDIER Sébastien suggestion in 
    [plotly forum](https://community.plotly.com/t/control-distance-between-stacked-bars/75303/4)

    Args:
        y_var (str): Name of the variable in `results`
        y_label (str): Label for the y axis

    Returns:
        go.Figure | None: Plotly figure object
    """

    n_plots = len(initial_states[system])

    y = []
    x = []
    facet_col = []
    for initial_state in initial_states[system]:
        for d in metadata[system]:
            # y.append( 
            #     get_nested_attr(d, f"n_paths_per_initial_state.{initial_state}")
            # )
            y.append(d["n_paths_per_initial_state"][initial_state]),
            x.append(f'N={d["n_horizon"]}'),
            facet_col.append(initial_state)

    # Prepare data
    data = dict(
        x=x,
        y = y,
        facet_col=facet_col,
        # 'color': alternatives,
    )
    # print(data)
    # data = pd.DataFrame()
    # print(data)

    fig = px.bar(data, x="x", y="y", facet_col="facet_col") # color="color", facet_col="facet_col", pattern_shape="color"

    # little customisations
    fig.for_each_annotation(lambda a: a.update(text=''))# remove the facet titles
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        barmode='group', 
        legend=dict(title='Color legend', x=0.8, y=1.3),
        # margin=dict(l=0, r=0, t=100, b=20),
        width = n_plots * 300, #  if width is None else width,
        title=f"Number of paths per initial state for <b>{system}</b>"
    )
    fig.update_yaxes(title_text=y_label, ticks="outside", linecolor="black", row=1, col=1)
    
    domains = domains_calculator(gap=0.05, n_plot=n_plots)

    [fig.update_xaxes(title_text=f"<b>{intial_state}</b>", domain=domains[idx], row=1, col=idx+1) for idx, intial_state in enumerate(initial_states[system])]
    [fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=idx+1) for idx in range(len(initial_states[system]))]

    return fig


def paths_per_initial_state_per_system_viz(metadata: dict, systems: list[str] = None) -> list[go.Figure]:
    """_summary_

    Args:
        metadata (dict): _description_
        systems (list[str], optional): _description_. Defaults to None.

    Returns:
        list[go.Figure]: _description_
    """

    if systems is None:
        systems = list(metadata.keys())
    
    states_per_system = {
        system: list(metadata[system][0]["n_paths_per_initial_state"].keys())
        for system in systems
    }
    
    initial_states = {
        system: states_per_system[system]
        for system in systems
    }
    
    # Flatten the metadata
    # data = [d for d in metadata[system] for system in systems]
    
    figs = [generate_n_paths_per_initial_state_bar_plot(metadata, system=system, initial_states=initial_states) for system in systems]
    
    return figs