from utils.constants import colors
from utils.constants import vars_info, colors_gnome
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly_resampler.figure_resampler import FigureWidgetResampler
from plotly.subplots import make_subplots

v = vars_info


def solar_med_visualization(results_df):
    
    nrows = 8
    height = 1500
    trace_idx = 0
    font_size=14

    color_blue = 'rgba(53, 132, 228, 0.5)'
    color_red  = 'rgba(224, 27, 36, 0.5)'
    
    r = results_df

    custom_data = np.stack((
        # Decision variables
        r['Tmed_s_in'].values, 
        r['Tmed_c_out'].values,
        r['mmed_s'].values,
        r['mmed_f'].values,
        r['Tsf_out'].values,
        r['mts_src'].values,
        
        # Environment variables
        r['Tamb'].values,
        r['Tmed_c_in'].values,
        r['wmed_f'].values,
        r['Tts_t_in'].values,
        r['mts_src'].values,    
    ), axis=-1)
    
    # Build hover text

    hover_text = f"""
    %{{x}}<br>
    %{{meta[0]}}: %{{y:.2f}}<br><br>

    <b>Decision variables</b><br>
    - {v['Tmed_s_in']['label_plotly']}: %{{customdata[0]:.1f}} {v['Tmed_s_in']['units_model']}<br>
    - {v['Tmed_c_out']['label_plotly']}: %{{customdata[1]:.1f}} {v['Tmed_c_out']['units_model']}<br>
    - {v['mmed_s']['label_plotly']}: %{{customdata[2]:.1f}} {v['mmed_s']['units_model']}<br>
    - {v['mmed_f']['label_plotly']}: %{{customdata[3]:.1f}} {v['mmed_f']['units_model']}<br><br>

    <b>Environment variables</b><br>
    - {v['Tamb']['label_plotly']}: %{{customdata[4]:.1f}} {v['Tamb']['units_model']}<br>
    - {v['Tmed_c_in']['label_plotly']}: %{{customdata[5]:.1f}} {v['Tmed_c_in']['units_model']}<br>
    - {v['wmed_f']['label_plotly']}: %{{customdata[6]:.1f}} {v['wmed_f']['units_model']}<br>
    - {v['Tts_t_in']['label_plotly']}: %{{customdata[7]:.1f}} {v['Tts_t_in']['units_model']}<br>
    - {v['mts_src']['label_plotly']}: %{{customdata[8]:.1f}} {v['mts_src']['units_model']}
    """
    

    # Wrap a figure with FigureWidgetResampler
    fw_fig = FigureWidgetResampler(make_subplots(rows=nrows, shared_xaxes=True,
                                                vertical_spacing=0.025,
                                                subplot_titles=['Thermal storage temperature evolution', 'Solar Field temperature evolution', 'Flows',
                                                                'Heat exchanger temperatures', 'MED flows', 'MED temperatures', 'System metrics', 'Environment'],
                                                specs=[[{"secondary_y": False}],
                                                        [{"secondary_y": False}],
                                                        [{"secondary_y": False}],
                                                        [{"secondary_y": False}],
                                                        [{"secondary_y": True}],
                                                        [{"secondary_y": True}],
                                                        [{"secondary_y": True}],
                                                        [{"secondary_y": True}]]),
                                #    config={'toImageButtonOptions': {
                                #                 'format': 'svg', # one of png, svg, jpeg, webp
                                #                 'filename': 'librescada-plot',
                                #             },}
                                                )
    trace_idx+=1

    # MED Hot water inlet temperature
    var_name = 'Tmed_s_in'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(go.Scattergl(name=label,
                                    meta=[label],
                                    line=dict(color=colors_gnome['greys'][3], dash='dash'),
                                    hovertemplate=hover_text, customdata=custom_data,
                                    legendgrouptitle=dict(text='Thermal storage'),
                                    legendgroup='1'), 
                        hf_x=x, hf_y=results_df[var_name], 
                        row=1, col=1)
    trace_idx+=1

    # Thermal storage
    for idx, tank in enumerate(['h','c']):
        row_idx=1
        for pos, dash in zip(['t','m','b'], ['solid', 'dash', 'dot']):
            var_name = f'Tts_{tank}_{pos}'
            label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
            fw_fig.add_trace(
                go.Scattergl(
                    name=label,
                    meta=[label],
                    line=dict(color=colors[idx], dash=dash),
                    hovertemplate=hover_text, customdata=custom_data,
                    #   legendgroup=f'{idx}'),
                    legendgroup='1'
                ), 
                hf_x=x, hf_y=results_df[var_name], 
                row=row_idx, col=1)
            trace_idx+=1

    # Solar field
    ## Temperatures
    row_idx = 2
    colors_ = [color_blue, color_red]
    for idx, Type in enumerate(['in','out']):
        var_name = f'Tsf_{Type}'
        label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
        fw_fig.add_trace(
            go.Scattergl(
                name=label,
                meta=[label],
                line=dict(color=colors_[idx]),#, dash=dash),
                hovertemplate=hover_text, customdata=custom_data,
                legendgroup='2',
                legendgrouptitle=dict(text='Solar field'),
            ), 
            hf_x=x, hf_y=results_df[var_name], 
            row=row_idx, col=1
        )
        trace_idx+=1

    ## Flows solar field and heat exchanger
    var_name = 'q_sf'
    row_idx = 3
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(
        go.Scattergl(
            name=label,
            meta=[label],
            line=dict(color=color_red, dash='solid'),
            hovertemplate=hover_text, customdata=custom_data,
            legendgroup='3',
            legendgrouptitle=dict(text='sf & ts'),
        ), 
        hf_x=x, hf_y=results_df['msf'], 
        row=row_idx, col=1)

    var_name = 'mts_src'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(
        go.Scattergl(
            name=label,
            meta=[label],
            line=dict(color=color_blue, dash='solid'),
            hovertemplate=hover_text, customdata=custom_data,
            legendgroup='3',
        ), 
        hf_x=x, hf_y=results_df[var_name], 
        row=row_idx, col=1)

    # Heat exchanger temperatures
    row_idx = 4
    var_ids = [['Tsf_out', 'Tsf_in'], ['Tts_h_t', 'Tts_c_b']]
    # var_ids = [['Thx_p_in', 'Thx_p_out'], ['Thx_s_out', 'Thx_s_in']]
    for var_id, color in zip(var_ids, [color_red, color_blue]):
        # Add filled area between Thx_p_in and Thx_p_out with reddish color
        fw_fig.add_trace(
            go.Scattergl(
                # fill='tonexty', 
                line=dict(color=color, width=2), 
                mode='lines', 
                fillcolor=color, 
                legendgroup='4',
                name=vars_info[var_id[0]].get("label_plotly", vars_info[var_id[0]]['label'])
            ),
            hf_x=x, hf_y=results_df[var_id[0]], 
            row=row_idx, col=1
        )
        fw_fig.add_trace(
            go.Scattergl(
                # fill='tonexty', 
                mode='lines',
                line=dict(color=color, width=2), 
                fillcolor=color,
                legendgroup='4',
                name=vars_info[var_id[1]].get("label_plotly", vars_info[var_id[1]]['label'])
            ),
            hf_x=x, hf_y=results_df[var_id[1]], 
            row=row_idx, col=1
        )

    ## Energy stored
    ### Temporary
    # var_name = 'mts_src'
    # label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    # fw_fig.add_trace(go.Scattergl(name=label,
    #                                 meta=[label],
    #                                 # line=dict(color=colors[idx]),#, dash=dash),
    #                                 hovertemplate=hover_text, customdata=custom_data,
    #                                 legendgroup='4'), 
    #                         hf_x=x, hf_y=results_df[var_name], 
    #                         row=4, col=1)
    # trace_idx+=1

    # fw_fig.add_trace(go.Scattergl(), hf_x=x, hf_y=results_df['Tts_c'], row=2, col=1)

    fw_fig.update_yaxes(title_text="(ºC)", row=1, col=1)
    # fw_fig.update_yaxes(title_text="Cold tank", row=2, col=1)
    fw_fig.update_yaxes(title_text="(ºC)", row=2, col=1)
    fw_fig.update_yaxes(title_text="(m³/h)", row=3, col=1)
    fw_fig.update_yaxes(title_text="(ºC)", row=4, col=1)
    # fw_fig.update_yaxes(title_text="Energy", row=4, col=1)
    # fw_fig.update_layout(yaxis1=dict(domain=[1, 1/nrows]), yaxis2=dict(domain=[1/nrows, 2/nrows]))

    # Solar field

    # MED
    row_idx = 5

    ## Flows
    ### Distillate
    var_name = 'mmed_d'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                    line=dict(color=colors_gnome['blues'][1], dash='dash'),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup='5',
                                legendgrouptitle=dict(text='MED'),), 
                            hf_x=x, hf_y=results_df[var_name], 
                            row=row_idx, col=1)

    ### Hot water

    var_name = 'mmed_s'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                #   line=dict(color=colors[idx], dash=dash),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup='5'), 
                            hf_x=x, hf_y=results_df[var_name], 
                            secondary_y=True,
                            row=row_idx, col=1)

    ### Feedwater
    var_name = 'mmed_f'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                #   line=dict(color=colors[idx], dash=dash),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup='5'), 
                                secondary_y=True,
                            hf_x=x, hf_y=results_df[var_name], 
                            row=row_idx, col=1)

    ### Cooling water
    var_name = 'mmed_c'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                #   line=dict(color=colors[idx], dash=dash),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup='5'), 
                                secondary_y=True,
                            hf_x=x, hf_y=results_df[var_name], 
                            row=row_idx, col=1)

    # # Update axis for distillate
    # fw_fig.data[trace_idx].update(yaxis=f"y{axis_idx}")
    # print(fw_fig.data[trace_idx]['name'])
    # # # Update axis for other flows
    # for data in fw_fig.data[trace_idx+1:trace_idx+4]:
    #     print(data['name'])
    #     data.update(yaxis=f"y{axis_idx+1}")
        
    ## Temperatures
    # axis_idx = 7
    # trace_idx = 14
    row_idx = 6

    for dash, name in zip(['solid', 'dash'], ['in', 'out']):
        var_name = f'Tmed_s_{name}'
        label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
        fw_fig.add_trace(go.Scattergl(name=label,
                                    meta=[label],
                                    line=dict(color=colors[0], dash=dash),
                                    hovertemplate=hover_text, customdata=custom_data,
                                    legendgroup=f'{row_idx}'), 
                                secondary_y=False,
                                hf_x=x, hf_y=results_df[var_name], 
                                row=row_idx, col=1)
        
    for dash, name in zip(['solid', 'dash'], ['in', 'out']):
        var_name = f'Tmed_c_{name}'
        label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
        fw_fig.add_trace(go.Scattergl(name=label,
                                    meta=[label],
                                    line=dict(color=colors[1], dash=dash),
                                    hovertemplate=hover_text, customdata=custom_data,
                                    legendgroup=f'{row_idx}'), 
                                secondary_y=True,
                                hf_x=x, hf_y=results_df[var_name], 
                                row=row_idx, col=1)

    # print('')
    # print([f"{i}: {data['name']}" for i, data in enumerate(fw_fig.data)])
    # print([f"{i}: {data['yaxis']}" for i, data in enumerate(fw_fig.data)])
    # # # Update axis for heat source
    # for data in fw_fig.data[trace_idx:trace_idx+2]:
    #     print(data['name'])
    #     data.update(yaxis=f"y{axis_idx}")
    # print('')
    # # Update axis for heat sink
    # for data in fw_fig.data[trace_idx+2:trace_idx+4]:
    #     print(data['name'])
    #     data.update(yaxis=f"y{axis_idx+1}")
        
    # print([f"{i}: {data['yaxis']}" for i, data in enumerate(fw_fig.data)])

    # # System
    row_idx = 7
    # axis_idx = 9
    # trace_idx = 18

    ## STEC
    var_name = 'STEC_med'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    label_stec = label
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                    line=dict(color=colors_gnome['oranges'][-2]),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup=f'{row_idx}',
                                legendgrouptitle=dict(text='Metrics'),), 
                    secondary_y=False,
                    hf_x=x, hf_y=results_df[var_name], 
                    row=row_idx, col=1)

    var_name = 'SEEC_med'
    label = v[var_name]['label'] + f" ({v[var_name]['units_model']})"
    label_sec = label
    fw_fig.add_trace(go.Scattergl(name=label,
                                meta=[label],
                                    line=dict(color=colors_gnome['purples'][-2]),
                                hovertemplate=hover_text, customdata=custom_data,
                                legendgroup=f'{row_idx}'), 
                    secondary_y=True,
                    hf_x=x, hf_y=results_df[var_name], 
                    row=row_idx, col=1)

    # var_name = 'SEC_sf'
    # label = "SEC<sub>sf</sub> (kWe/kWth)"
    # label_sec = label
    # fw_fig.add_trace(go.Scattergl(name=label,
    #                               meta=[label],
    #                                 line=dict(color=colors_gnome['purples'][-4]),
    #                               hovertemplate=hover_text, customdata=custom_data,
    #                               legendgroup=f'{row_idx}'), 
    #                  secondary_y=True,
    #                  hf_x=x, hf_y=results_df[var_name], 
    #                  row=row_idx, col=1)

    # var_name = 'SEC_sf'
    # label = "SEC<sub>sf</sub> (kWe/kWth)"
    # label_sec = label
    # fw_fig.add_trace(go.Scattergl(name=label,
    #                               meta=[label],
    #                                 line=dict(color=colors_gnome['purples'][-4]),
    #                               hovertemplate=hover_text, customdata=custom_data,
    #                               legendgroup=f'{row_idx}'), 
    #                  secondary_y=True,
    #                  hf_x=x, hf_y=results_df[var_name], 
    #                  row=row_idx, col=1)

    # fw_fig.update_yaxes(title_text=label, row=1, col=1)

    # in order for autoshift to work, you need to set x-anchor to free

    # MED temperature axis
    span = np.max([ np.abs(np.max(results_df['Tmed_s_in'])- np.min(results_df['Tmed_s_out'])), 
                np.abs(np.max(results_df['Tmed_c_in'])- np.min(results_df['Tmed_c_out'])) ]) * 1.2

    Ts_avg = np.mean([np.max(results_df['Tmed_s_in']), np.min(results_df['Tmed_s_out'])])
    Tc_avg = np.mean([np.max(results_df['Tmed_c_in']), np.min(results_df['Tmed_c_out'])])

    fw_fig.update_layout(
        # xaxis2=dict(domain=[0, 1], anchor="y2"),
        yaxis5=dict(
            title="mmed_d",
            titlefont=dict(color=colors_gnome['blues'][1]),
            tickfont=dict(color=colors_gnome['blues'][1]),
        ),
        # yaxis5=dict(
        #     title="mmed_d",
        #     titlefont=dict(color="#d62728"),
        #     tickfont=dict(color="#d62728"),
        # ),
        yaxis7=dict(
            title="Heat source",
            titlefont=dict(color=colors[0]),
            tickfont=dict(color=colors[0]),
            range=[Ts_avg-span/2, Ts_avg+span/2],
            dtick=span/5
        ),
        yaxis8=dict(
            title="Heat sink",
            titlefont=dict(color=colors[1]),
            tickfont=dict(color=colors[1]),
            range=[Tc_avg-span/2, Tc_avg+span/2],
            dtick=span/5
        ),
        yaxis9=dict(
            title=label_stec,
            titlefont=dict(color=colors_gnome['oranges'][-2]),
            tickfont=dict(color=colors_gnome['oranges'][-2]),
            range=[30, 120],
            # dtick=span/5
        ),
        yaxis10=dict(
            title=label_sec,
            titlefont=dict(color=colors_gnome['purples'][-2]),
            tickfont=dict(color=colors_gnome['purples'][-2]),
            range=[1, 20],
            # dtick=span/5
        ),
    )

    # Environment variables
    row_idx = 8
    var_id = 'Tamb'
    label = 'T<sub>amb</sub>'
    fw_fig.add_trace(
        go.Scattergl(
            name=label,
            meta=[label],        
            legendgroup=str(row_idx),
            legendgrouptitle=dict(text='Environment variables'),    
        ), 
        secondary_y=False,
        hf_x=x, hf_y=results_df[var_id],
        row=row_idx, col=1
    )
    fw_fig.update_yaxes(title_text=vars_info[var_id]["units_model"], row=row_idx, col=1)

    var_id = 'I'
    label = 'I'
    fw_fig.add_trace(
        go.Scattergl(
            name=label,
            meta=[label],        
            legendgroup=str(row_idx),
        ), 
        secondary_y=True,
        hf_x=x, hf_y=results_df[var_id],
        row=row_idx, col=1
    )
    fw_fig.update_yaxes(title_text=vars_info[var_id]["units_model"], row=row_idx, col=1)

    fw_fig.update_layout(
        yaxis_title=vars_info['Tamb']["units_model"],
        yaxis2_title=vars_info['I']["units_model"]
    )
    fw_fig.update_xaxes(tickfont=dict(size=font_size), row=row_idx, col=1)
    # Update Y-axis settings for both subplots
    fw_fig.update_yaxes(title_font=dict(size=font_size), tickfont=dict(size=font_size), row=row_idx, col=1)

    height = 1000
    # Figure layout
    fw_fig.update_layout(height=height, showlegend=True, 
                        title='Simulation results',
                        margin=dict(t=50, b=50, l=50, r=50),
                        hovermode='closest',
                        #  legend_tracegroupgap=height/nrows-60,
                        newshape_line_color=colors_gnome['purples'][1],
                        modebar_add=['drawline', 'drawopenpath', 'drawcircle', 'drawrect','eraseshape'])

    # print([f"{i}: {data['name']}" for i, data in enumerate(fw_fig.data)])
    # print([f"{i}: {data['yaxis']}" for i, data in enumerate(fw_fig.data)])
    
    return fw_fig
