from solarMED_modeling import SupportedStatesType, MedState

def convert_to_state(state: str, state_cls: SupportedStatesType = MedState) -> SupportedStatesType:
    return getattr(state_cls, state)