import plotly.graph_objs as go
import pandas as pd
import plotly

from models_psa import MedState, SF_TS_State

# plotly.colors.qualitative

colors = {
    "gray": "#E2E2E2",
    "yellow": "#EECA3B",
    "purple": "#A349A4",
    "green": "#7DBE3C",
    "white": "#FFFFFF",
    "transparent": "rgba(0,0,0,0)"

}


def state_evolution_plot(df, iteration: int = None):

    # Encode states to integers

    if 'state_sf_ts' not in df.columns:
        raise ValueError("Column 'state_sf_ts' not found in the DataFrame")
    if 'state_med' not in df.columns:
        raise ValueError("Column 'state_med' not found in the DataFrame")

    if not isinstance(df["state_sf_ts"][0], SF_TS_State):
        raise ValueError("The column 'state_sf_ts' must be of type SF_TS_State")

    if not isinstance(df["state_med"][0], MedState):
        raise ValueError("The column 'state_med' must be of type MedState")

    # sf_active = False and (ts_active = False or ts_active=True) -> 1
    # sf_active = True and ts_active = False -> 6
    # sf_active = True and ts_active = True -> 11

    df["sf_ts_value"] = (1.5 *  (df["state_sf_ts"] == SF_TS_State.IDLE) +
                         1.5 *  (df["state_sf_ts"] == SF_TS_State.RECIRCULATING_TS) +
                         6.5 *  (df["state_sf_ts"] == SF_TS_State.HEATING_UP_SF) +
                         11.5 * (df["state_sf_ts"] == SF_TS_State.SF_HEATING_TS))

    # Assign the following labels to the states:
    # 1 -> "S̲F̲ T̲S̲"
    # 6 -> "S̅F̅ T̲S̲"
    # 11 -> "S̅F̅ T̅S̅"

    df["sf_ts_label"] = df["sf_ts_value"].map({1.5: "S̲F̲ T̲S̲", 6.5: "S̅F̅ T̲S̲", 11.5: "S̅F̅ T̅S̅"})
    df["sf_ts_bg"] = df["sf_ts_value"].map({1.5: colors["gray"], 6.5: colors["white"], 11.5: colors["yellow"]})
    df["sf_ts_border"] = df["sf_ts_value"].map(
        {1.5: colors["transparent"], 6.5: colors["yellow"], 11.5: colors["yellow"]}
    )

    # state_med = INACTIVE -> 3
    # state_med = STARTING -> 8
    # state_med = ACTIVE -> 13
    # state_med = STOPPING -> 8

    df["state_med_value"] = (2.5 *  (df["state_med"] == MedState.OFF) +
                             7.5 *  (df["state_med"] == MedState.GENERATING_VACUUM) +
                             7.5 *  (df["state_med"] == MedState.IDLE) +
                             7.5 *  (df["state_med"] == MedState.STARTING_UP) +
                             7.5 *  (df["state_med"] == MedState.SHUTTING_DOWN) +
                             12.5 * (df["state_med"] == MedState.ACTIVE))

    # Assign the following labels to the states:
    # 3 -> "M̲E̲D̲"
    # 8 -> "M̲E̲D̲"
    # 13 -> "M̅E̅D̅"

    df["state_med_label"] = df["state_med_value"].map({2.5: "M̲E̲D̲", 7.5: "M̲E̲D̲", 12.5: "M̅E̅D̅"})
    df["state_med_bg"] = df["state_med_value"].map({2.5: colors["gray"], 7.5: colors["white"], 12.5: colors["green"]})
    df["state_med_border"] = df["state_med_value"].map(
        {2.5: colors["transparent"], 7.5: colors["green"], 12.5: colors["green"]}
    )

    # Create a figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["sf_ts_value"],
        mode="lines+markers",
        name="SF+TS state",
        line=dict(color=colors["yellow"], width=0.5),
        text=df["sf_ts_label"],
        textposition="middle center",
        # marker=dict(
        #     size=15,
        #     color="red",
        #     symbol="circle",
        #     line=dict(
        #         width=10,
        #         color='DarkSlateGrey'
        #     )
        # )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["state_med_value"],
        mode="lines",
        name="state_med_value",
        line=dict(color=colors["purple"], width=0.5),
    ))

    # Update layout
    annotations = []

    for i, row in df.iterrows():
        annotations.append(dict(
            x=row.name,
            y=row['sf_ts_value'],
            text=row['sf_ts_label'],
            showarrow=False,
            bordercolor=row['sf_ts_border'],
            borderwidth=3,
            bgcolor=row['sf_ts_bg'],
            # Bold text
            # font=dict(weight="bold")
        ))
        annotations.append(dict(
            x=row.name,
            y=row['state_med_value'],
            text=row['state_med_label'],
            showarrow=False,
            bordercolor=row['state_med_border'],
            borderwidth=3,
            bgcolor=row['state_med_bg'],
        ))

    # Add a vertical bar in iteration if provided
    if iteration:
        fig.add_shape(
            dict(
                type="rect", # other options are "rect", "circle", "line"
                x0=iteration-0.5,
                x1=iteration+0.5,
                y0=0,
                y1=15,
                # line=dict(
                #     color="orange",
                #     width=1,
                #     dash="dashdot"
                # )
                fillcolor="rgba(182, 182, 182,0.1)",
            )
        )

    fig.update_layout(
        annotations=annotations,
        title="<b>State Machine evolution<b>" if iteration is None else f"<b>State Machine evolution</b> - Iteration {iteration}<br>{df['state_title'][iteration]}</br>",
        xaxis_title="Time",
        yaxis_title="State",
        legend_title="Legend",
        showlegend=False,
        # Add some margin to the right
        margin=dict(r=10, t=70, b=10, l=10),
        # plot_bgcolor="rgba(182, 182, 182,0.1)",
        # plot_bgcolor="rgba(182, 182, 182,0.15)",
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            tickvals=[2, 7, 12],
            tickcolor=colors["gray"],
            ticktext=["Idle", "Transition", "Active"],
            # minor_ticks="inside"
            griddash='dash',
            minor=dict(
                tickvals=[5, 9],
                showgrid=True,
                gridwidth=10,
                griddash='solid',
            ),
            # Remove zero line
            zeroline=False,
        ),
        xaxis=dict(zeroline=False),
    )

    return fig


def costs_plot():
    pass


def med_relative_cons_plot():
    pass