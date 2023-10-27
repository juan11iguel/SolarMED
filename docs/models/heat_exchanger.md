# Heat exchanger model
## Fuente

[1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico, “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy, vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.

[2] J. A. Duffie and W. A. Beckman, Solar engineering of thermal processes. John Wiley & Sons, 2013.


### Nomenclature

![center | solarMED_optimization-Heat_exchanger | 500x500](attachments/solarMED_optimization-Heat_exchanger.svg)

Loops (L):
- $p:$ Primary (hot sink / side) loop
- $s:$ Secondary (cold sink / side) loop 

- $T_{hx,L,in} \: (\degree C):$ Inlet temperature
- $T_{hx,L,out} \: (\degree C):$ Outlet temperature
- $\dot{m}_L\: (m^3/h?):$  Flow rate
- $P_{gen} \: (kW_{th}):$ Power supplied by the hot side
- $P_{abs} \: (kW_{th}):$ Power transferred to the cold side
- $\eta_{hx}:$ Exchange efficiency  
- $T_{amb} \: (\degree C):$ Ambient temperature

### Inputs / outputs

$$ T_{hx,p,out},T_{hx,s,out},P_{gen},P_{abs} = f(T_{hx,p,in},T_{hx,s,in},\dot{m}_p,\dot{m}_{s},(UA)_{hx}) $$
$$  T_{hx,p,out},T_{hx,s,out} = f(T_{hx,p,in},T_{hx,s,in},\dot{m}_p,\dot{m}_{s},(UA)_{hx})  $$
$UA$ is a parameter to be calibrated. It depends on the heat exchange surface (if known it can just be substituted) and the heat transfer coefficient, which depends on the temperature difference between the ambient and heat exchanger. GA (genetic algorithms) approaches will be used.
- Por defecto $(UA)_{hx}=28000 \: [\frac{W}{\degree C}]$

## Equations

$$
\begin{equation} \tag{1}
T_{hx,p,out} = T_{hx,p,in}-\eta_{hx,p}·\frac{(\dot{m}c_p)_{min}}{\dot{m}_{p}c_{p,p}}(T_{hx,p,in}-T_{hx,s,in})
\end{equation}
$$
$$
\begin{equation} \tag{2}
\: T_{hx,s,out} = T_{hx,s,in}+\eta_{hx,s}·(T_{hx,p,in}-T_{hx,p,out})
\end{equation}
$$
$$
\begin{equation} \tag{3}
\eta_{hx,p}=\frac{1-e^{\theta}}{1-\frac{(\dot{m}c_{p})_{min}}{(\dot{m}c_{p})_{max}}e^{\theta}}
\end{equation}
$$
$$
\begin{equation} \tag{4}
\eta_{hx,s}=\frac{\dot{m}_{p}c_{p,p}}{\dot{m}_{s}c_{p,s}}
\end{equation}
$$
$$
\begin{equation} \tag{5}
\theta = UA·\left( \frac{1}{(\dot{m}c_{p})_{max}} - \frac{1}{(\dot{m}c_{p})_{min}} \right)
\end{equation}
$$
$$
P_{gen} = \dot{m}_p·c_{p,p}·(T_{hx,p,in}-T_{hx,p,out})
$$
$$
P_{abs} = \dot{m}_s·c_{p,s}·(T_{hx,s,out}-T_{hx,s,in})
$$

## Otros

![Intercambiador campo](../Validation%20signals%20list.md#Intercambiador%20campo)