import numpy as np

def estimate_flow_ts_discharge(qdis: float, Tdis_in: float, Tdis_out: float, Tsrc: float, upper_limit_m3h: float = 54) -> float:
    """
    By using the three-way valve and knowing the valve discharge flow and temperatures, and the inlet temperature
    Nomenclature from docs (docs/models/three_way_valve.md):
    - $T_{src} \: (\degree C):$ Temperature from source (thermal storage)
    - $T_{dis,in} \: (\degree C):$ Inlet temperature to discharge (MED heat source inlet, $T_{med,s,in}$)
    - $T_{dis,out} \: (\degree C):$ Outlet temperature from discharge (MED heat source outlet, $T_{med,s,out}$)
    - $\dot{m}_{dis} \: (kg/s):$  Flow rate through load / discharge
    - $\dot{m}_{src} \: (kg/s)=(1-R)·\dot{m}_{dis} :$  Flow rate from source
    """
    qsrc = qdis * (Tdis_in - Tdis_out) / (Tsrc - Tdis_out)

    return np.min( [np.max([qsrc, 0]), upper_limit_m3h] )