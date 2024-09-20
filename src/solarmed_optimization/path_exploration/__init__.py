from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from solarmed_modeling import MedState, SF_TS_State, SolarMED_State, SolarMedState_with_value, SfTsState_with_value
from solarmed_modeling.fsms import SolarFieldWithThermalStorage_FSM, MedFSM

SupportedStates = MedState | SF_TS_State | SolarMED_State
SupportedFSMs = MedFSM | SolarFieldWithThermalStorage_FSM

class Node(BaseModel):
    """
    Node model. Barebones class just to validate the nodes are being generated correctly
    """
    step_idx: int
    state: SupportedStates

    # Derived fields
    node_id: str = None
    state_value: str | int = None
    state_name: str = None
    x_pos: float =  None
    y_pos: float = None

    model_config = ConfigDict(
        validate_assignment=False,
    )

    def model_post_init(self, __context):
        self.node_id = f'step{self.step_idx:03d}_{self.state.value}'
        self.state_value = str(self.state.value)
        self.state_name = self.state.name
        self.x_pos = self.step_idx

        if isinstance(self.state, SolarMED_State):
            self.y_pos = getattr(SolarMedState_with_value, self.state.name).value
        elif isinstance(self.state, SF_TS_State):
            self.y_pos = getattr(SfTsState_with_value, self.state.name).value
        else:
            self.y_pos = float(self.state.value)

class Edge(BaseModel):
    """
    Edge model. Barebones class just to validate the edges are being generated correctly
    """
    step_idx: int
    src_name: str
    dst_name: str
    # src_node_id: int
    # dst_node_id: int
    src_node_id: str
    dst_node_id: str
    transition_id: str
    line_type: str = 'solid' # Solid line for direct transitions and dashed for in-progress transitions
    x_pos_src: float = None
    x_pos_dst: float = None
    y_pos_src: float = None
    y_pos_dst: float = None

    model_config = ConfigDict(
        validate_assignment=False,
    )

    def model_post_init(self, __context):
        self.line_type = self.line_type.lower()


def generate_edges(result_list: list[dict], step_idx: int, system: Literal['MED', 'SFTS'], Np: int, ) -> list[dict]:

    """
    Generate edges for the given FSM

    Aspects to consider:
        - By naming convention, long transitions are triggered with a transition/trigger id that starts with 'start_',
        and finishes with 'finish_'. When a transition is in progress, the trigger id is 'inprogress_' and is blocked
        from being triggered again until the transition is completed.

    :param result_list:
    :param step_idx:
    :param subsystem:
    :return:
    """

    machine_init_args = dict(
        sample_time = 1,

    )

    if system.lower() == 'sfts':
        states = [state for state in SF_TS_State]
        machine_cls = SolarFieldWithThermalStorage_FSM
        state_cls = SF_TS_State

    elif system.lower() == 'med':
        states = [state for state in MedState]
        machine_cls = MedFSM
        state_cls = MedState

        machine_init_args.update(
            vacuum_duration_time=5,
            brine_emptying_time=2,
            startup_duration_time=2
        )

    else:
        raise NotImplementedError(f'Unsupported subsystem {system}')

    for state in states:
        model = machine_cls(**machine_init_args)

        triggers = model.machine.get_triggers(state)
        for trigger in triggers:

            # print(f'{state.name} -> {trigger}')
            dst_state = model.machine.get_transitions(trigger)[0].dest
            dst_state = getattr(state_cls, dst_state)

            # By default state transitions are "instantaneous"
            duration = 1

            if state == MedState.STARTING_UP:
                duration = model.startup_duration_samples

            elif state == MedState.GENERATING_VACUUM:
                duration = model.vacuum_duration_samples

            elif state == MedState.SHUTTING_DOWN:
                duration = model.brine_emptying_samples

            # Cancel transitions are instantaneous
            if 'cancel' in trigger: duration = 1

            for idx in range(step_idx, step_idx + duration - 1):
                if idx + 1 < Np:
                    result_list.append(
                        Edge(
                            step_idx=idx,
                            src_name=state.name,
                            dst_name=state.name,
                            # src_node_id=int(f'{idx}{state.value}'),
                            # dst_node_id=int(f'{idx + 1}{state.value}'),
                            src_node_id=f'step{idx:03d}_{state.value}',
                            dst_node_id=f'step{idx + 1:03d}_{state.value}',
                            line_type='dash',
                            transition_id=f'step{idx:03d}_' + trigger.replace('finish',
                                                                              'inprogress') + f'_since_step{step_idx}',
                        ).model_dump()
                    )

            if step_idx + duration < Np:
                # In the last step, the transition is completed
                result_list.append(
                    Edge(
                        step_idx=step_idx + duration - 1,
                        src_name=state.name,
                        dst_name=dst_state.name,
                        # src_node_id=int(f'{step_idx + duration - 1}{state.value}'),
                        # dst_node_id=int(f'{step_idx + duration}{dst_state.value}'),
                        src_node_id=f'step{step_idx + duration - 1:03d}_{state.value}',
                        dst_node_id=f'step{step_idx + duration:03d}_{dst_state.value}',
                        transition_id=f'step{step_idx + duration - 1:03d}_' + trigger if duration == 1 else f'step{step_idx + duration - 1:03d}_{trigger}_from_step{step_idx}',
                        line_type='solid'
                    ).model_dump()
                )

        if step_idx + 1 < Np:
            # Always add the option to remain in the same state
            result_list.append(
                Edge(
                    step_idx=step_idx,
                    src_name=state.name,
                    dst_name=state.name,
                    # src_node_id=int(f'{step_idx}{state.value}'),
                    # dst_node_id=int(f'{step_idx + 1}{state.value}'),
                    src_node_id=f'step{step_idx:03d}_{state.value}',
                    dst_node_id=f'step{step_idx + 1:03d}_{state.value}',
                    transition_id=f'step{step_idx:03d}_none',
                    line_type='solid'
                ).model_dump()
            )

    return result_list


def generate_edges_dataframe(edges_list: list[dict], nodes_df: pd.DataFrame = None) -> pd.DataFrame:

    df = pd.DataFrame(edges_list)

    # Make sure the first column is the src_node_id and the second is the dst_node_id
    cols = ['src_node_id', 'dst_node_id'] + [col for col in df.columns if col not in ['src_node_id', 'dst_node_id']]
    df = df.reindex(columns=cols)

    if nodes_df is not None:
        generate_edges_coordinates(nodes_df, edges_df=df)

    return df

def generate_edges_coordinates(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """
    Update the edges DataFrame with the x and y positions of the source and destination nodes

    :param nodes_df:
    :param edges_df:
    :return:
    """

    # Not needed, already initialized in the Edge model
    # edges_df[
    #     ['x_pos_src', 'x_pos_dst', 'y_pos_src', 'y_pos_dst']
    # ] = pd.DataFrame(np.zeros((len(edges_df), 4)), index=edges_df.index)

    for idx, row in edges_df.iterrows():
        # Find every row in the edges DataFrame that has the same src_node_name as the current node

        src_node_name = row['src_node_id']
        dst_node_name = row['dst_node_id']

        src_node = nodes_df[nodes_df['node_id'] == src_node_name]
        dst_node = nodes_df[nodes_df['node_id'] == dst_node_name]

        if len(src_node) > 1 or len(dst_node) > 1:
            raise RuntimeError(f"Multiple nodes with the same name {src_node_name} / {dst_node_name} found")

        if len(src_node) == 0 or len(dst_node) == 0:
            raise RuntimeError(f"No nodes with the name {src_node_name} / {dst_node_name} found")

        src_node = src_node.iloc[0]
        dst_node = dst_node.iloc[0]

        increment = 0.01 * src_node['x_pos'] if row['line_type'] == 'dash' else 0

        if dst_node['x_pos'] - src_node['x_pos'] > 1:
            raise RuntimeError(f"More than one transition duration from {src_node_name} to {dst_node_name}")

        edges_df.loc[idx, 'x_pos_src'] = src_node['x_pos']
        edges_df.loc[idx, 'x_pos_dst'] = dst_node['x_pos'] - 0.1  # So the arrow is not on top of the node circle
        edges_df.loc[idx, 'y_pos_src'] = src_node['y_pos'] + increment
        edges_df.loc[idx, 'y_pos_dst'] = dst_node['y_pos'] + increment