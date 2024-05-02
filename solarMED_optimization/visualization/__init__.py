from typing import Literal
from enum import Enum
import numpy as np
import pandas as pd
import copy

import plotly.graph_objs as go

from solarMED_modeling import SupportedStatesType, MedState, SF_TS_State, SolarMED_State, SolarMedState_with_value, SfTsState_with_value


node_colors = {
    str(MedState.__name__): "#c061cb", # Cuidado de no quitar el str() porque sino no se modifica la propia clase
    str(SF_TS_State.__name__): "#ff7800",
    str(SolarMED_State.__name__): '#6959CD'
}

# def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
#     L=len(pos)
#     if len(text)!=L:
#         raise ValueError('The lists pos and text must have the same len')
#     annotations = []
#     for k in range(L):
#         annotations.append(
#             dict(
#                 text=labels[k], # or replace labels with a different list for the text within the circle
#                 x=pos[k][0], y=2*M-position[k][1],
#                 xref='x1', yref='y1',
#                 font=dict(color=font_color, size=font_size),
#                 showarrow=False)
#         )
#     return annotations

# all_paths: list[list[SupportedStatesType]]

def plot_state_graph(nodes_df: pd.DataFrame | list[pd.DataFrame], system: Literal['MED', 'SFTS', 'SolarMED'], Np: int,
                     edges_df: pd.DataFrame = None, width=1400, height=500, title: str = None, results_df: pd.DataFrame = None) -> go.FigureWidget:

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
        state_cls = SF_TS_State
        state_cls_with_value = SfTsState_with_value
    elif system == 'SolarMED':
        state_cls = SolarMED_State
        state_cls_with_value = SolarMedState_with_value
    else:
        raise ValueError(f"System {system} and nodes_df {type(nodes_df)} not supported")

    if title is None:
        title = f"Directed graph of the operating modes evolution in the {state_cls.__name__} system"

    # Work with lists of nodes_df
    if not isinstance(nodes_df, list):
        nodes_df = [nodes_df]

    # Build vectors in the format wanted by plotly ([xsrc_0, xdst_0, None, xsrc_1, xdst_1, None, ...] and [ysrc_0, ydst_0, None, ysrc_1, ydst_1, None, ...])
    # Xe = []
    Xe_solid = []
    Xe_dash = []

    # Ye = []
    Ye_solid = []
    Ye_dash = []

    if edges_df is not None:
        for idx, row in edges_df.iterrows():
            # Xe += [row['x_pos_src'], row['x_pos_dst'], None]
            # Ye += [row['y_pos_src'], row['y_pos_dst'], None]

            Xe_solid += [row['x_pos_src'], row['x_pos_dst'], None] if row['line_type'] == 'solid' else []
            Ye_solid += [row['y_pos_src'], row['y_pos_dst'], None] if row['line_type'] == 'solid' else []

            Xe_dash += [row['x_pos_src'], row['x_pos_dst'], None] if row['line_type'] == 'dash' else []
            Ye_dash += [row['y_pos_src'], row['y_pos_dst'], None] if row['line_type'] == 'dash' else []


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
    for system_idx, n_df in enumerate(nodes_df):

        if type(n_df['state'][0]) == SolarMED_State:
            system_types.append(SolarMED_State)
            system_types_with_value.append(SolarMedState_with_value)
        elif type(n_df['state'][0]) == MedState:
            system_types.append(MedState)
            system_types_with_value.append(MedState)
        elif type(n_df['state'][0]) == SF_TS_State:
            system_types.append(SF_TS_State)
            system_types_with_value.append(SfTsState_with_value)
        else:
            raise ValueError(f"State {n_df['state'][0]} not supported")


        if results_df is not None:
            """
                Generate a path of coordinates for each subsystem given it's state
            """
            Xr.append([])
            Yr.append([])

            # Not generic, should be improved
            state_col = 'med_state' if system_types[-1] == MedState else 'sf_ts_state'

            for idx in range(len(results_df)-1):
                x_aux, y_aux = get_coordinates_edge(src_node_id=f"step{idx:03}_{results_df.iloc[idx][state_col].value}", # med_state, sfts_state, etc, cómo saber cuál es el adecuado?
                                                    dst_node_id=f"step{idx+1:03}_{results_df.iloc[idx+1][state_col].value}",
                                                    nodes_df=n_df, y_shift=last_val)
                Xr[system_idx] += x_aux
                Yr[system_idx] += y_aux

        fig.add_trace(
            go.Scatter(
                x=n_df['x_pos'].values,
                y=n_df['y_pos'].values + last_val,
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


    # Create separate traces for solid and dashed lines
    fig.add_trace(
        go.Scatter(
            x=Xe_solid,
            y=Ye_solid,
            mode='lines+markers',
            line=dict(color='rgb(210,210,210)', width=1, dash='solid'),
            marker=dict(symbol='arrow', size=10, color='rgb(210,210,210)', angleref="previous"),
            text=edges_df['transition_id'].values if edges_df is not None else None,
            hoverinfo='text',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=Xe_dash,
            y=Ye_dash,
            mode='lines+markers',
            line=dict(color='rgb(210,210,210)', width=1, dash='dash'),
            marker=dict(symbol='arrow', size=10, color='rgb(210,210,210)', angleref="previous"),
            hoverinfo='none'
        )
    )

    # Empty scatter to be used for highlighting arriving edges. Also re-using to represent results paths, if there are more than
    # two subsystems this won't work, it should done apart in a loop
    fig.add_trace(
        go.Scatter(
            x=Xr[0] if len(Xr) > 0 else None, y=Yr[0] if len(Yr) > 0 else None,
            hoverinfo='none',
            mode='lines+markers',
            line=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
                                            width=3, dash='solid'),
            marker=dict(symbol='arrow', size=10, color='#06C892', angleref="previous"),
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
            marker=dict(symbol='arrow', size=10, color='#AD72F3', angleref="previous"),
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
        tickvals=np.arange(Np, step=1),
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
            title='Time steps'
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



