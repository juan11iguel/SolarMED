from dataclasses import dataclass


@dataclass
class FixedModelParameters:
    """
    MED fixed model parameters.
    Default values could be replaced by values from a configuration file. 
    """
    Tmed_s_min: float = 60  # Minimum operating heat source temperature [ºC]
    Tmed_s_max: float = 75  # Maximum operating heat source temperature [ºC]
    qmed_c_min: float = 8   # Minimum cooling flow rate [m³/h]
    qmed_c_max: float = 21  # Maximum cooling flow rate [m³/h]
    qmed_s_min: float = 30  # Minimum heat source flow rate [m³/h]
    qmed_s_max: float = 48  # Maximum heat source flow rate [m³/h]
    qmed_f_min: float = 5   # Minimum feed flow rate [m³/h]
    qmed_f_max: float = 9   # Maximum feed flow rate [m³/h]