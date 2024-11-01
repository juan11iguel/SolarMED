from typing import Literal
import numpy as np
import pandas as pd
import math
from loguru import logger

import plotly.graph_objs as go

from solarmed_modeling import SupportedStatesType, MedState, SfTsState, SolarMedState, SolarMedState_with_value, SfTsState_with_value
from . import Node, generate_edges, generate_edges_dataframe


node_colors = {
    str(MedState.__name__): "#c061cb", # Cuidado de no quitar el str() porque sino no se modifica la propia clase
    str(SfTsState.__name__): "#ff7800",
    str(SolarMedState.__name__): '#6959CD'
}


def plot_state_graph(nodes_df: pd.DataFrame | list[pd.DataFrame], system: Literal['MED', 'SFTS', 'SolarMED'], Np: int,
                     edges_df: pd.DataFrame | list[pd.DataFrame] = None, width=1400, height=500,
                     title: str = None, results_df: pd.DataFrame = None, max_samples: int = 30,
                     highligth_step: int = None) -> go.FigureWidget:

    """

    :param all_paths:
    :param nodes_df:
    :param edges_df:
    :return:
    """


    if system == 'MED' and isinstance(nodes_df, pd.DataFrame):
        state_cls = MedState
        state_cls_with_value = state_cls
    elif system == 'SFTS' and isinstance(nodes_df, pd.DataFrame):
        state_cls = SfTsState
        state_cls_with_value = SfTsState_with_value
    elif system == 'SolarMED':
        state_cls = SolarMedState
        state_cls_with_value = SolarMedState_with_value
    else:
        raise ValueError(f"System {system} and nodes_df {type(nodes_df)} not supported")

    if title is None:
        title = f"Directed graph of the operating modes evolution in the {state_cls.__name__} system"


    if (isinstance(nodes_df, pd.DataFrame) and isinstance(edges_df, list) or
            isinstance(nodes_df, list) and isinstance(edges_df, pd.DataFrame)):
        raise ValueError("If nodes_df is a list, then edges_df should be a list too and viceversa")

    # Work with lists of both nodes_df and edges_df
    if not isinstance(nodes_df, list):
        nodes_df = [nodes_df]
    if not isinstance(edges_df, list) and edges_df is not None:
        edges_df = [edges_df]

    # Step size given max_samples
    step_size = max(1, math.ceil(len(results_df) / max_samples))

    if step_size > 1:
        logger.warning(f'There are more samples than the maximum specified ({max_samples}), states will be shown every {step_size} samples. Aliasing may occur')
        # From [jmmease](https://community.plotly.com/t/change-title-color-using-html/18217/2):
        # > plotly.js doesn’t support arbitrary HTML markup in labels. Here is a description of the subset that is
        # supported https://help.plot.ly/adding-HTML-and-links-to-charts/#step-2-the-essentials
        title += f'<br><span style="font-size: 11px; font-color:"orange">(States shown every {step_size} samples, aliasing is likely to occur)</span></br>'

    fig = go.FigureWidget()

    # fig.add_trace(
    #     go.Scatter(
    #         x=Xe,
    #         y=Ye,
    #         mode='lines',
    #         line= dict(color='rgb(210,210,210)', width=1, dash=edges_df['line_type'].values.tolist()),
    #         hoverinfo='none'
    #     )
    # )

    system_types = []
    system_types_with_value = []
    ticktext = []
    tickvals = []
    last_val = 0
    Xr = []
    Yr = []
    Xr_dash = []
    Yr_dash = []
    # Xe = []
    Xe_solid = []
    Xe_dash = []

    # Ye = []
    Ye_solid = []
    Ye_dash = []
    for system_idx, n_df in enumerate(nodes_df):

        if type(n_df['state'][0]) == SolarMedState:
            system_types.append(SolarMedState)
            system_types_with_value.append(SolarMedState_with_value)
        elif type(n_df['state'][0]) == MedState:
            system_types.append(MedState)
            system_types_with_value.append(MedState)
        elif type(n_df['state'][0]) == SfTsState:
            system_types.append(SfTsState)
            system_types_with_value.append(SfTsState_with_value)
        else:
            raise ValueError(f"State {n_df['state'][0]} not supported")


        if results_df is not None:
            """
                Generate a path of coordinates for each subsystem given it's state
            """
            Xr.append([])
            Yr.append([])
            Xr_dash.append([])
            Yr_dash.append([])

            # TODO: Not generic, should be improved
            state_col = 'med_state' if system_types[-1] == MedState else 'sf_ts_state'

            for idx in range(0, len(results_df)-step_size, step_size):

                x_aux, y_aux = get_coordinates_edge(src_node_id=f"step{idx:03}_{results_df.iloc[idx][state_col].value}",
                                                    dst_node_id=f"step{idx+step_size:03}_{results_df.iloc[idx+step_size][state_col].value}",
                                                    nodes_df=n_df, y_shift=last_val)

                # If the y value of the current state is equal from the previous one, add a solid line to connect them
                if results_df.iloc[idx][state_col].value == results_df.iloc[idx+step_size][state_col].value:
                    Xr[system_idx] += x_aux
                    Yr[system_idx] += y_aux

                # If the y value of the current state is different from the previous one, add a dahsed line to connect them
                else:
                    Xr_dash[system_idx] += x_aux
                    Yr_dash[system_idx] += y_aux

                # If highligth_step is not None, create a circle around the node
                if highligth_step is not None and idx == highligth_step:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_aux[0]], y=[y_aux[0]],
                            mode='markers',
                            marker=dict(symbol='circle-dot', size=40, color="#f5c211", ),
                            line=None,
                        )
                    )
            # Terrible, just for the last that was not included in the loop:
            if highligth_step is not None and idx == highligth_step:
                fig.add_trace(
                    go.Scatter(
                        x=[x_aux[0]], y=[y_aux[0]],
                        mode='markers',
                        marker=dict(symbol='circle-dot', size=40, color="#f5c211", ),
                        line=None,
                    )
                )


        # Add result paths
        if edges_df is not None:
            """
                Generate the coordinates for the possible paths for each subsystem provided in its edges_df
            """

            Xe_solid.append([])
            Ye_solid.append([])
            Xe_dash.append([])
            Ye_dash.append([])

            for idx in range(0, len(edges_df[system_idx]), step_size):
                row = edges_df[system_idx].iloc[idx]
                # Build vectors in the format wanted by plotly ([xsrc_0, xdst_0, None, xsrc_1, xdst_1, None, ...] and [ysrc_0, ydst_0, None, ysrc_1, ydst_1, None, ...])
                # Xe += [row['x_pos_src'], row['x_pos_dst'], None]
                # Ye += [row['y_pos_src'], row['y_pos_dst'], None]

                Xe_solid[system_idx] += [row['x_pos_src'], row['x_pos_dst'], None] if row['line_type'] == 'solid' else []
                Ye_solid[system_idx] += [row['y_pos_src'] + last_val, row['y_pos_dst'] + last_val, None] if row['line_type'] == 'solid' else []

                Xe_dash[system_idx] += [row['x_pos_src'], row['x_pos_dst'], None] if row['line_type'] == 'dash' else []
                Ye_dash[system_idx] += [row['y_pos_src'] + last_val, row['y_pos_dst'] + last_val, None] if row['line_type'] == 'dash' else []

        # Add edges
        if edges_df is not None:
            # Create separate traces for solid and dashed lines
            fig.add_trace(
                go.Scatter(
                    x=Xe_solid[system_idx],
                    y=Ye_solid[system_idx],
                    mode='lines+markers',
                    line=dict(color='rgb(210,210,210)', width=1, dash='solid'),
                    marker=dict(symbol='arrow', size=10, color='rgb(210,210,210)', angleref="previous"),
                    text=edges_df[system_idx]['transition_id'].values,
                    hoverinfo='text',
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=Xe_dash[system_idx],
                    y=Ye_dash[system_idx],
                    mode='lines+markers',
                    line=dict(color='rgb(210,210,210)', width=1, dash='dash'),
                    marker=dict(symbol='arrow', size=10, color='rgb(210,210,210)', angleref="previous"),
                    hoverinfo='none'
                )
            )

        # Add nodes
        # Group the dataframe by 'step_idx'
        grouped_nodes = n_df.groupby('step_idx')

        # Filter the groups
        displayed_nodes = {name: group for name, group in grouped_nodes if name % step_size == 0}
        x_pos = [node_group['x_pos'].values for node_group in displayed_nodes.values()]
        y_pos = [node_group['y_pos'].values + last_val for node_group in displayed_nodes.values()]

        x_pos = np.concatenate(x_pos).tolist()
        y_pos = np.concatenate(y_pos).tolist()

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers',
                name='states',
                marker=dict(symbol='circle-dot', size=20, color=node_colors[system_types[-1].__name__]),
                line=dict(color='rgb(50,50,50)', width=0.5),
                text=n_df['state_name'].values,
                hoverinfo='text'
            )
        )

        # for system_ in system_types_with_value:
        ticktext += [state.name for state in system_types_with_value[-1]]
        tickvals += [state.value + last_val for state in system_types_with_value[-1]]

        last_val += len(system_types_with_value[-1])
        # last_val += len([state for state in system_types[-1]])


    # Empty scatter to be used for highlighting arriving edges. Also re-using to represent results paths, if there are more than
    # two subsystems this won't work, it should done apart in a loop
    fig.add_trace(
        go.Scatter(
            x=Xr[0] if len(Xr) > 0 else None, y=Yr[0] if len(Yr) > 0 else None,
            hoverinfo='none',
            mode='lines+markers',
            line=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
                                            width=3, dash='solid'),
            # marker=dict(symbol='arrow', size=10, color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
            #             angleref="previous"),
        )
    )
    # Empty scatter to be used for highlighting departing edges
    fig.add_trace(
        go.Scatter(
            x=Xr[-1] if len(Xr) > 1 else None, y=Yr[-1] if len(Yr) > 1 else None,
            hoverinfo='none',
            mode='lines+markers',
            line=dict(color='#AD72F3' if len(system_types) == 1 else node_colors[system_types[1].__name__],
                                            width=3, dash='solid'),
            # marker=dict(symbol='arrow', size=10, color='#AD72F3' if len(system_types) == 1 else node_colors[system_types[1].__name__]
            #             , angleref="previous"),
        )
    )

    # Add scatter for result dashed lines
    for idx in range(len(Xr_dash)):
        fig.add_trace(
            go.Scatter(
                x=Xr_dash[idx],
                y=Yr_dash[idx],
                hoverinfo='none',
                mode='lines+markers',
                line=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[idx].__name__],
                                            width=3, dash='dash'),
                # marker=dict(symbol='arrow', size=10, color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
                #             angleref="previous"),
            )
        )

    axis_conf = dict(
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        title=''
    )

    # if not isinstance(nodes_df, list):
    #     ticktext = [state.name for state in state_cls]
    #     tickvals = [state.value for state in state_cls_with_value]
    # else:

    fig.update_yaxes(
        ticktext=ticktext,
        tickvals=tickvals,
    )

    fig.update_xaxes(
        tickvals=np.arange(Np, step=step_size),
    )

    fig.update_layout(
        title=title,# + \
              # f"<br> Average number of alternative paths per state {options_avg}</br>",
        # "<br> Data source: <a href='https://networkdata.ics.uci.edu/data.php?id=11'> [1]</a>",
        font=dict(size=12),
        showlegend=False,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(
            zeroline=False,
            showline=False,
            showgrid=False,
            showticklabels=True,
            title=f'Samples (sample rate: {str(results_df.index.to_series().diff().mode()[0])})' if results_df is not None else 'Samples',
        ),
        yaxis=axis_conf,
        margin=dict(
            l=40,
            r=40,
            b=85,
            t=100,
        ),
        hovermode='closest',
        # annotations=[
        #    dict(
        #        showarrow=False,
        #         text='This igraph.Graph has the Kamada-Kawai layout',
        #         xref='paper', yref='paper',
        #         x=0, y=-0.1,
        #         xanchor='left', yanchor='bottom',
        #         font=dict(size=14)
        #     )
        # ]
    )

    return fig


# nodes_scatter.on_click(highlight_node_paths)

def get_coordinates(node_id: str, edges_df: pd.DataFrame, type: Literal['src', 'dst']) -> tuple[list[float], list[float]]:

    node_type = "dst" if type == "dst" else "src"

    edges = edges_df[edges_df[f'{node_type}_node_name'] == node_id]

    x_src_aux = edges['x_pos_src'].values
    x_dst_aux = edges['x_pos_dst'].values
    y_src_aux = edges['y_pos_src'].values
    y_dst_aux = edges['y_pos_dst'].values

    x = []
    y = []
    for xsrc, xdst, ysrc, ydst in zip(x_src_aux, x_dst_aux, y_src_aux, y_dst_aux):
        x += [xsrc, xdst, None]
        y += [ysrc, ydst, None]

    return x, y


def get_coordinates_edge(src_node_id: str, dst_node_id: str, nodes_df: pd.DataFrame, y_shift=0) -> tuple[list[float], list[float]]:

    src_node = nodes_df[nodes_df['node_id'] == src_node_id]
    dst_node = nodes_df[nodes_df['node_id'] == dst_node_id]

    if len(src_node) > 1 or len(dst_node) > 1:
        raise RuntimeError(f"Multiple nodes with the same name {src_node_id} / {dst_node_id} found")
    elif len(src_node) == 0 or len(dst_node) == 0:
        raise RuntimeError(f"No nodes with the name {src_node_id} / {dst_node_id} found")

    return (
        [src_node['x_pos'].values[0], dst_node['x_pos'].values[0], None],
        [src_node['y_pos'].values[0] + y_shift, dst_node['y_pos'].values[0]  + y_shift, None]
    )


def plot_episode_state_evolution(df: pd.DataFrame, subsystems_state_cls: SupportedStatesType, show_edges: bool = False,
                                 highligth_step: int = None, width: int = None, height: int = None) -> go.Figure | go.FigureWidget:

    Np = len(df)
    edges_df = None
    edges_list = []

    nodes_dfs = []
    edges_df = [] if show_edges else None

    for subsystem_cls in subsystems_state_cls:
        nodes_dfs.append(pd.DataFrame([
            Node(step_idx=step_idx, state=state).model_dump()
            for step_idx in range(Np) for state in [state for state in subsystem_cls]
        ]))

        # Generate edges dataframes
        if show_edges:
            if subsystem_cls == MedState:
                system = 'MED'
            elif subsystem_cls == SfTsState:
                system = 'SFTS'
            else:
                raise NotImplementedError(f'Unsupported subsystem {subsystem_cls}')

            for step_idx in range(Np):
                edges_list = generate_edges(edges_list, step_idx, system=system, Np=Np)

            edges_df.append( generate_edges_dataframe(edges_list, nodes_df=nodes_dfs[-1]) )

    # # Generate nodes dataframes
    # nodes_sfts_df = pd.DataFrame([
    #     Node(step_idx=step_idx, state=state).model_dump()
    #     for step_idx in range(Np) for state in [state for state in SfTsState]
    # ])
    # nodes_med_df = pd.DataFrame([
    #     Node(step_idx=step_idx, state=state).model_dump()
    #     for step_idx in range(Np) for state in [state for state in MedState]
    # ])

    fig = plot_state_graph(
        nodes_df=nodes_dfs,
        system='SolarMED',
        edges_df=edges_df,
        results_df=df,
        Np=Np,
        height=800 if height is None else height,
        width=1200 if width is None else width,
        highligth_step=highligth_step
    )

    return fig


