**Pendiente homogeneizar nomenclatura**

![center | solarMED_optimization-Complete system model.drawio](solarMED_optimization-Complete%20system%20model.drawio.svg)

### Nomenclatura

![solarMED_optimization-general_diagram](solarMED_optimization-general_diagram.svg)

- MED 
  - $T\_{med,s,in}$
  - $T\_{med,cw,out}$
  - $\\dot{m}\_{med,s}$
  - $\\dot{m}\_{med,f}$
- Solar field 
  - $T\_{sf,out}$
  - $\\dot{m}\_{sf}$
- Thermal storage 
  - $T\_{ts,t,in}$
  - $T\_{ts,t}$
  - $T\_{ts,b}$
  - 

### Inputs / outputs

$$ \\dot{m}*{med,d},SEEC*{med},STEC\_{MED},SEC\_{sf} = f(last_state, current_inputs,) $$
$UA$ is a parameter to be calibrated. It depends on the heat exchange surface (if known it can just be substituted) and the heat transfer coefficient, which depends on the temperature difference between the ambient and heat exchanger. GA (genetic algorithms) approaches will be used.