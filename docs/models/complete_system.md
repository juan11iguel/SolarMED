# Integrated system model

## Inputs / outputs flow diagram

![center | solarMED_optimization-Complete system model.drawio](attachments/solarMED_optimization-Complete%20system%20model.drawio.svg)

### Nomenclature

![solarMED_optimization-general_diagram](attachments/solarMED_optimization-general_diagram.svg)

- MED 
  - $T_{med,s,in}$
  - $T_{med,cw,out}$
  - $\dot{m}_{med,s}$
  - $\dot{m}_{med,f}$
- Solar field 
  - $T_{sf,out}$
  - $\dot{m}_{sf}$
- Thermal storage 
  - $T_{ts,t,in}$
  - $T_{ts,t}$
  - $T_{ts,b}$
  - 

### Inputs / outputs

$$ \dot{m}_{med,d},SEEC_{med},STEC_{MED},SEC_{sf} = f(last_state, current_inputs,) $$

$UA$ is a parameter to be calibrated. It depends on the heat exchange surface (if known it can just be substituted) and the heat transfer coefficient, which depends on the temperature difference between the ambient and heat exchanger. GA (genetic algorithms) approaches will be used.


# Results

![SolarMED_validation_20231030](../attachments/SolarMED_validation_20231030.png)