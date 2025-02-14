import copy
from dataclasses import asdict, fields
import numpy as np
import pandas as pd
import math
from enum import Enum
from loguru import logger
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from solarmed_modeling.fsms import (MedState, 
                                    SfTsState, 
                                    SolarMedState)
from solarmed_modeling.fsms.utils import (SupportedSystemsStatesType, 
                                          SupportedFSMTypes,
                                          SupportedSystemsStatesMapping,
                                          FsmInputsMapping)
                                        # (SupportedSystemsLiteral,
                                        # SupportedSystemsStatesMapping,)
                     
from . import Node, generate_edges, generate_edges_dataframe, get_coordinates_edge

node_colors = {
    str(MedState.__name__): "#c061cb", # Cuidado de no quitar el str() porque sino no se modifica la propia clase
    str(SfTsState.__name__): "#ff7800",
    str(SolarMedState.__name__): '#6959CD'
}


def plot_state_graph(nodes_df: pd.DataFrame | list[pd.DataFrame], Np: int, system_title: str = None,
                     edges_df: pd.DataFrame | list[pd.DataFrame] = None, width=1400, height=500,
                     title: str = None, results_df: pd.DataFrame = None, max_samples: int = 30,
                     highligth_step: int = None, state_cols: list[str] | str = None,
                     subtitle: str = None, show_inputs_subplot: bool = False) -> go.FigureWidget:

    """

    :param all_paths:
    :param nodes_df:
    :param edges_df:
    :return:
    """
    # Validation
    if edges_df is not None:
        assert type(nodes_df) is type(edges_df), "nodes_df and edges_df should both be lists (with same number of elements) or dataframes"
    
    if state_cols is not None:
        if not isinstance(state_cols, list):
            state_cols = [state_cols]
    
        assert len(nodes_df) == len(state_cols), "The number of state_cols should be the same as the number of subsystems"
    # state_cls = getattr(SupportedSystemsStatesMapping, system).value

    # TODO: This is not generic, should be improved
    # if system == 'MED' and isinstance(nodes_df, pd.DataFrame):
    #     state_cls = MedState
    #     state_cls_with_value = state_cls
    # elif system == 'SFTS' and isinstance(nodes_df, pd.DataFrame):
    #     state_cls = SfTsState
    #     state_cls_with_value = SfTsState_with_value
    # elif system == 'SolarMED':
    #     state_cls = SolarMedState
    #     state_cls_with_value = SolarMedState_with_value
    # else:
    #     raise ValueError(f"System {system} and nodes_df {type(nodes_df)} not supported")

    if title is None:
        title = f"Directed graph of the operating modes evolution. {system_title}"

    # Work with lists of both nodes_df and edges_df
    if not isinstance(nodes_df, list):
        nodes_df = [nodes_df]
    if not isinstance(edges_df, list) and edges_df is not None:
        edges_df = [edges_df]
        
    # Infer system types keys from state column of nodes_df
    system_type_keys: list[str] = [SupportedSystemsStatesMapping(type(node_df.iloc[0]["state"])).name for node_df in nodes_df]

    # Step size given max_samples
    step_size = 1
    if results_df is not None:
        step_size = max(step_size, math.ceil(len(results_df) / max_samples))

    if step_size > 1:
        logger.warning(f'There are more samples than the maximum specified ({max_samples}), states will be shown every {step_size} samples. Aliasing may occur')
        # From [jmmease](https://community.plotly.com/t/change-title-color-using-html/18217/2):
        # > plotly.js doesn’t support arbitrary HTML markup in labels. Here is a description of the subset that is
        # supported https://help.plot.ly/adding-HTML-and-links-to-charts/#step-2-the-essentials
        title += f'<br><span style="font-size: 11px; font-color:"orange">(States shown every {step_size} samples, aliasing is likely to occur)</span></br>'

    xrange = np.arange(Np, step=step_size)

    if not show_inputs_subplot:
        fig = make_subplots(rows=1, cols=1)
    else:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[1/4, 3/4],
            shared_xaxes=True,
            vertical_spacing=0.02
        )

    # fig.add_trace(
    #     go.Scatter(
    #         x=Xe,
    #         y=Ye,
    #         mode='lines',
    #         line= dict(color='rgb(210,210,210)', width=1, dash=edges_df['line_type'].values.tolist()),
    #         hoverinfo='none'
    #     )
    # )
    
    # Add empty scatter for valid inputs if enabled
    if show_inputs_subplot:
        fig.update_yaxes(title="Inputs", row=1, col=1)
        
        for system_key in system_type_keys:
            inputs_enum = FsmInputsMapping[system_key].value
            for field in fields(inputs_enum):
                fig.add_trace(
                    go.Scatter(
                        name=f"{field.name}", # [{field.type}]
                        x=xrange,
                        y=results_df.iloc[xrange][field.name].values.astype(float) if results_df is not None else None,
                        hoverinfo='name+x+y',
                        stackgroup='inputs',
                        showlegend=True,
                        legendgroup="inputs",
                    ),
                    row=1, col=1
                )


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

        # TODO: This is not generic, should be improved
        # if type(n_df['state'][0]) == SolarMedState:
        #     system_types.append(SolarMedState)
        #     system_types_with_value.append(SolarMedState_with_value)
        # elif type(n_df['state'][0]) == MedState:
        #     system_types.append(MedState)
        #     system_types_with_value.append(MedState)
        # elif type(n_df['state'][0]) == SfTsState:
        #     system_types.append(SfTsState)
        #     system_types_with_value.append(SfTsState_with_value)
        # else:
        #     raise ValueError(f"State {n_df['state'][0]} not supported")
        state_cls_ = type(n_df['state'][0])
        system_types.append( state_cls_ )
        state_cls_with_value = Enum('state_with_value', {
            f'{state.name}': i
            for i, state in enumerate(state_cls_)
        })
        system_types_with_value.append(state_cls_with_value)


        if results_df is not None:
            """
                Generate a path of coordinates for each subsystem given it's state
            """
            Xr.append([])
            Yr.append([])
            Xr_dash.append([])
            Yr_dash.append([])

            # TODO: Not generic, should be improved
            if state_cols is not None:
                state_col = state_cols[system_idx]
            else:
                state_col = 'med_state' if system_types[-1] == MedState else 'sf_ts_state'
            # Chapuza para tapar chapuza
            if state_col not in results_df.columns:
                assert "state" in results_df, f"Neither {state_col}, nor `state` are present in the results dataframe, input a valid `state_col` with the system states"
                state_col = "state"

            for idx in range(0, len(results_df)-step_size, step_size):

                x_aux, y_aux = get_coordinates_edge(src_node_id=f"step{idx:03}_{int(results_df.iloc[idx][state_col])}",
                                                    dst_node_id=f"step{idx+step_size:03}_{int(results_df.iloc[idx+step_size][state_col])}",
                                                    nodes_df=n_df, y_shift=last_val)

                # If the y value of the current state is equal from the previous one, add a solid line to connect them
                if int(results_df.iloc[idx][state_col]) == int(results_df.iloc[idx+step_size][state_col]):
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
                            showlegend=False,
                        ),
                        row=1 if not show_inputs_subplot else 2, col=1,
                    )
            # Terrible, just for the last that was not included in the loop:
            if highligth_step is not None and idx == highligth_step:
                fig.add_trace(
                    go.Scatter(
                        x=[x_aux[0]], y=[y_aux[0]],
                        mode='markers',
                        marker=dict(symbol='circle-dot', size=40, color="#f5c211", ),
                        line=None,
                        showlegend=False,
                    ),
                    row=1 if not show_inputs_subplot else 2, col=1,
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
                    showlegend=False,
                ),
                row=1 if not show_inputs_subplot else 2, col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=Xe_dash[system_idx],
                    y=Ye_dash[system_idx],
                    mode='lines+markers',
                    line=dict(color='rgb(210,210,210)', width=1, dash='dash'),
                    marker=dict(symbol='arrow', size=10, color='rgb(210,210,210)', angleref="previous"),
                    hoverinfo='none',
                    showlegend=False,
                ),
                row=1 if not show_inputs_subplot else 2, col=1,
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
                hoverinfo='text',
                showlegend=False,
            ),
            row=1 if not show_inputs_subplot else 2, col=1,
        )

        # for system_ in system_types_with_value:
        ticktext += [f"{state.name} —" for state in system_types_with_value[-1]]
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
            marker=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
                        size=16),
            showlegend=False,
            # marker=dict(symbol='arrow', size=10, color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
            #             angleref="previous"),
        ),
        row=1 if not show_inputs_subplot else 2, col=1,
    )
    # Empty scatter to be used for highlighting departing edges
    fig.add_trace(
        go.Scatter(
            x=Xr[-1] if len(Xr) > 1 else None, y=Yr[-1] if len(Yr) > 1 else None,
            hoverinfo='none',
            mode='lines+markers',
            showlegend=False,
            line=dict(color='#AD72F3' if len(system_types) == 1 else node_colors[system_types[1].__name__],
                                            width=3, dash='solid'),
            # marker=dict(symbol='arrow', size=10, color='#AD72F3' if len(system_types) == 1 else node_colors[system_types[1].__name__]
            #             , angleref="previous"),
        ),
        row=1 if not show_inputs_subplot else 2, col=1,
    )

    # Add scatter for result dashed lines
    for idx in range(len(Xr_dash)):
        fig.add_trace(
            go.Scatter(
                x=Xr_dash[idx],
                y=Yr_dash[idx],
                hoverinfo='none',
                mode='lines+markers',
                showlegend=False,
                line=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[idx].__name__],
                                            width=3, dash='dash'),
                marker=dict(color='#06C892' if len(system_types) == 1 else node_colors[system_types[idx].__name__],
                        size=16)
                # marker=dict(symbol='arrow', size=10, color='#06C892' if len(system_types) == 1 else node_colors[system_types[0].__name__],
                #             angleref="previous"),
            ),
            row=1 if not show_inputs_subplot else 2, col=1,
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
        showticklabels=True,
        # showtickmarkers=True,
        **axis_conf,
        row=1 if not show_inputs_subplot else 2, col=1,
    )

    fig.update_xaxes(
        tickvals=xrange,
        zeroline=False,
        showline=False,
        showgrid=False,
        showticklabels=True,
        row=1 if not show_inputs_subplot else 2, col=1,
    )

    xaxis_title = 'Samples'
    if results_df is not None:
        try:
            xaxis_title =f'Samples (sample rate: {str(results_df.index.to_series().diff().mode()[0])})'
        except Exception:
            pass
        
    fig.update_layout(
        font=dict(size=12),
        title=dict(
            text=title,
            # + \
            # f"<br> Average number of alternative paths per state {options_avg}</br>",
            # "<br> Data source: <a href='https://networkdata.ics.uci.edu/data.php?id=11'> [1]</a>",
            subtitle=dict(
                text=subtitle if subtitle is not None else "",
                font=dict(color="gray", size=11),
            ),
            x=0,
            xanchor= 'left',
        ),
        legend=dict(
            x=1,  # Position on x-axis (right-most)
            y=1.05,  # Position above the plot area
            xanchor='right',  # Anchor legend box to the right
            yanchor='bottom',  # Anchor legend box to the bottom
            traceorder='normal',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        showlegend=True if show_inputs_subplot else False,
        autosize=False,
        width=width,
        height=height,
        # yaxis=axis_conf,
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
    
    axis_id = "xaxis" if not show_inputs_subplot else "xaxis2"
    fig.layout.update(
        {axis_id: dict(title=xaxis_title)}
    )
    
    # for data in fig.data:
    #     data.xaxis = 'x'

    return go.FigureWidget(fig)


def plot_episode_state_evolution(df: pd.DataFrame, subsystems_state_cls: list[SupportedSystemsStatesType] = None, show_edges: bool = False,
                                 highligth_step: int = None, width: int = None, height: int = None, show_inputs_subplot: bool = False,
                                 title: str = None, subtitle: str = None, model: SupportedFSMTypes = None) -> go.Figure | go.FigureWidget:

    """
        If the model instance is provided, some additional information will be added to the plot
    """

    assert not(subsystems_state_cls is None and model is None), "Either `subsystems_state_cls` or `model` should be provided"

    Np = len(df)
    edges_df = None
    edges_list = []

    nodes_dfs = []
    edges_df = [] if show_edges else None
    
    if subsystems_state_cls is None:
        subsystems_state_cls = model._state_type
        
    if not isinstance(subsystems_state_cls, list):
        subsystems_state_cls = [subsystems_state_cls]

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
    
    if model is not None:
        params: dict = asdict(model.params)
        init_int_sts = asdict(model.initial_internal_state)
        params_enum = ''.join( [f"<br>{key.replace('_', ' ')}: {value}" for key, value in params.items() if isinstance(value, (int, float, bool, str))] )
        init_int_sts_enum = ''.join([f"<br>{key.replace('_', ' ')}: {value}" for key, value in init_int_sts.items() if not isinstance(value, bool)])

        # Build custom title and subtitle
        if title is None:
            title = f"<b>{model.name}</b> <br>State evolution from: <br>{model.initial_state.name.replace('_', ' ')}"
        if subtitle is None:
            subtitle = f"""<b>Parameters</b> {params_enum} <br><br><b>Initial internal conditions</b> {init_int_sts_enum}"""
        # Build custom hover information
        labels = [key for key, value in init_int_sts.items() if isinstance(value, (int, float, bool, str))]
        hovertemplate = "(sample: %{x}, state: <b>%{customdata[0]}</b>) <br><br><b>Internal states</b>" 
        hovertemplate = hovertemplate + ''.join([f"<br>{label.replace('_', ' ')}: %{{customdata[{idx+1}]}}" for idx, label in enumerate(labels)]) + "<extra></extra>"


    fig = plot_state_graph(
        nodes_df=nodes_dfs,
        system_title='SolarMED',
        edges_df=edges_df,
        results_df=df,
        Np=Np,
        height=800 if height is None else height,
        width=1200 if width is None else width,
        highligth_step=highligth_step,
        title=title,
        subtitle=subtitle,
        show_inputs_subplot=show_inputs_subplot,
    )

    if model is not None:
        fig.update_yaxes(side="right")
        # title_x = 0.2
        # base_title_x = 0.2
        # base_left_margin = 220
        # left_margin_proportion = 
        # left_margin = title_x * 
        fig.update_layout(
            margin=dict(t=50, b=0, pad=1, l=220, r=20),
            title_yanchor="top",
            title_y=0.8,
            title_xanchor="right",
            title_x=0.2,
            height=400 if height is None else height,
            width=1000 if width is None else width,
            hovermode='x unified',
        )

        # Add custom hover information
        """ The hovers are not working completely, I suspect is because they are
        being attached to the scatter of states, which may have a strange data shape, 
        maybe to the line traces it would.
        
        Confirmed by changing hovermode to x, and the hover changes by changing the y value
        and not the x.
        
        -> Add an invisible scatter trace to add the custom hover information"""
        customdata = copy.deepcopy(df[["state"] + labels])
        customdata["state"] = customdata["state"].apply(lambda x: x.name)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=np.zeros(len(df)),
                mode='markers',
                marker=dict(color='rgba(0, 0, 0, 0)'),
                hovertemplate=hovertemplate,
                customdata=customdata.values,
            )
        )
        fig.update_traces(
            hoverinfo='none',
            selector=(({"name": "states"}))
        )

    return fig