import plotly.graph_objs as go
import pandas as pd
from enum import Enum
import plotly

from solarMED_optimization import MedState, SF_TS_State

# plotly.colors.qualitative

colors = {
    "gray": "#E2E2E2",
    "yellow": "#EECA3B",
    "purple": "#A349A4",
    "green": "#7DBE3C",
    "white": "#FFFFFF",
    "transparent": "rgba(0,0,0,0)"

}

def state_evolution_plot(df, ):

    # Encode states to integers

    if 'sf_ts_state' not in df.columns:
        raise ValueError("Column 'sf_ts_state' not found in the DataFrame")
    if 'med_state' not in df.columns:
        raise ValueError("Column 'med_state' not found in the DataFrame")

    # sf_active = False and (ts_active = False or ts_active=True) -> 1
    # sf_active = True and ts_active = False -> 6
    # sf_active = True and ts_active = True -> 11

    df["sf_ts_value"] = (1.5 *  (df["sf_ts_state"] == SF_TS_State.IDLE) +
                         1.5 *  (df["sf_ts_state"] == SF_TS_State.RECIRCULATING_TS) +
                         6.5 *  (df["sf_ts_state"] == SF_TS_State.HEATING_UP_SF) +
                         11.5 * (df["sf_ts_state"] == SF_TS_State.SF_HEATING_TS))

    # Assign the following labels to the states:
    # 1 -> "S̲F̲ T̲S̲"
    # 6 -> "S̅F̅ T̲S̲"
    # 11 -> "S̅F̅ T̅S̅"

    df["sf_ts_label"] = df["sf_ts_value"].map({1.5: "S̲F̲ T̲S̲", 6.5: "S̅F̅ T̲S̲", 11.5: "S̅F̅ T̅S̅"})
    df["sf_ts_bg"] = df["sf_ts_value"].map({1.5: colors["gray"], 6.5: colors["white"], 11.5: colors["yellow"]})
    df["sf_ts_border"] = df["sf_ts_value"].map(
        {1.5: colors["transparent"], 6.5: colors["yellow"], 11.5: colors["yellow"]}
    )

    # med_state = INACTIVE -> 3
    # med_state = STARTING -> 8
    # med_state = ACTIVE -> 13
    # med_state = STOPPING -> 8

    df["med_state_value"] = (2.5 *  (df["med_state"] == MedState.OFF) +
                             7.5 *  (df["med_state"] == MedState.IDLE) +
                             7.5 *  (df["med_state"] == MedState.STARTING_UP) +
                             7.5 *  (df["med_state"] == MedState.SHUTTING_DOWN) +
                             12.5 * (df["med_state"] == MedState.ACTIVE))

    # Assign the following labels to the states:
    # 3 -> "M̲E̲D̲"
    # 8 -> "M̲E̲D̲"
    # 13 -> "M̅E̅D̅"

    df["med_state_label"] = df["med_state_value"].map({2.5: "M̲E̲D̲", 7.5: "M̲E̲D̲", 12.5: "M̅E̅D̅"})
    df["med_state_bg"] = df["med_state_value"].map({2.5: colors["gray"], 7.5: colors["white"], 12.5: colors["green"]})
    df["med_state_border"] = df["med_state_value"].map(
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
        y=df["med_state_value"],
        mode="lines",
        name="med_state_value",
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
            y=row['med_state_value'],
            text=row['med_state_label'],
            showarrow=False,
            bordercolor=row['med_state_border'],
            borderwidth=3,
            bgcolor=row['med_state_bg'],
        ))

    fig.update_layout(
        annotations=annotations,
        title="State Machine",
        xaxis_title="Time",
        yaxis_title="State",
        legend_title="Legend",
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