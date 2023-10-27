El campo solar es básicamente un conversor de energía eléctrica a térmica, el factor de conversión es el SEC ($kW_{e}/kW_{th}$), si es menor de uno quiere decir que tenemos que dar al sistema más energía eléctrica que la térmica que obtenemos, valores por encima de uno significa que hay una ganancia. Pero no sólo importa el factor de intercambio si no también a qué temperatura se obtiene el calor (exergía).

En el diagrama se muestran los lazos individuales que conforman el campo, en condiciones de estacionario y para el modelo se considera que los caudales y temperaturas son iguales en cada uno de los  lazos, y por lo tanto se puede simplificar a un único lazo con área de captador la suma de los lazos individuales.

![center | solarMED_optimization-solar_field.drawio](solarMED_optimization-solar_field.drawio.svg)

## Fuente

[1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico, “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy, vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.
[2] G. Ampuño, L. Roca, J. D. Gil, M. Berenguel, and J. E. Normey-Rico, “Apparent delay analysis for a flat-plate solar field model designed for control purposes,” Solar Energy, vol. 177, pp. 241–254, Jan. 2019, doi: 10.1016/j.solener.2018.11.014.

### Nomenclatura

- $T_{in} : (\degree C):$ Inlet temperature
- $T_{out} : (\degree C):$ Outlet temperature
- $\dot{m}_{sf}: (kg/s):$  Flow rate
- $P_{gen} : (kW_{th}):$ Thermal power generated
- $I : \left( \frac{W}{m^{2}} \right):$ Ambient temperature
- $T_{amb} : (\degree C):$ Ambient temperature
- $n_p:$ Number of parallel collectors in each loop
- $n_s:$ Number of serial connections of collectors rows
- $n_t:$ Number of parallel tubes in each collector
- $L_t:$ Length of the collector inner tube
- $H : \left( \frac{J}{s·\degree C} \right):$ Thermal losses coefficient for the loop
- $\beta : (m):$ Irradiance model parameter??
- $c_f:$ Conversion factor, used to add collectors depending on the configuration
- $A_{cs}:(m^{2}):$Flat plate collector tube cross-section area

### Inputs / outputs

$$ \dot{m}_{sf},P_{gen},SEC_{sf} = f(T_{in},T_{out,k}, T_{out,k-1}, I, T_{amb}) $$
$$ \dot{m}_{sf} = f(T_{in},T_{out}, I, T_{amb}) $$
Parameters: $H,\beta,n_t,n_p,n_s,L_t,A_{cs}$

$H$ and $\beta$ are dynamic parameters to be calibrated, the rest are constant and characteristic of the solar field. $H$ is a thermal loss coefficient, which depends on the temperature difference between the ambient and the system average temperature (approx.). $\beta$ la verdad que no lo tengo muy claro todavía, imagino que es el que se ve potenciado por los espejos @Lidia. GA (genetic algorithms) approaches will be used.

Por defecto:

- $H=2.2 : [J·s^{-1}·\degree C^{-1}]$
- $\beta=0.0975 : [m]$
- $n_p=7$
- $n_t=50$
- $n_s=2$
- $L_{t}=1.94 : [m]$
- $c_f=5*7*6·10^{5} : [sL·min^-1·m^{-3}]$
- $A_{cs}=7.85·10^{-5}:[m^2]$

## Equations

$$
\begin{equation} \tag{1}
\rho·c_p·A_{cs}·\frac{\delta T_{out}(t)}{\delta t}=\beta I(d_I) - \frac{H}{L_{eq}}(\tilde{T}(t)-T_{amb}(t)) - \frac{c_{p}}{c_{f}}· \frac{1}{L_{eq}}·\dot{m}_{sf}(t-d*\dot{m})·(T_{out}(t)-T_{in}(t-d_{Tin}))
\end{equation}
$$
$$
\begin{equation} \tag{2}
\tilde{T}(t) = \frac{T_{out}(t)+T_{in}(t-d_{Tin})}{2}
\end{equation}
$$
$$ L_{eq}=n_s·L_t $$
$$ c_f=n_p·n_t $$

Desarrollando en estacionario, $\frac{\delta T_{out}(t)}{\delta t}=0$, retardos no se consideran y despejando $\dot{m_{sf}}$:
$$
\begin{equation} \tag{3}
\dot{m}_{sf} =\frac{ + \beta I - \frac{H}{L_{eq}}(\tilde{T}-T_{amb}) }{  \frac{c_{f}}{c_{p,\tilde{T}}}· L_{eq}·(T_{out}-T_{in}) }
\end{equation}
$$
$$
\begin{equation} \tag{4}
\tilde{T}(k) = \frac{T_{out}(k)+T_{in}(k)}{2}
\end{equation}
$$

**Nuevo antiguo**. Realmente en este caso no nos importa la temperatura de salida anterior, se asume que el período que transcurre entre una evaluación y la siguiente es suficientemente grande para que la temperatura anterior sea irrelevante a la hora de determinar el caudal (tiene sentido, la temperatura anterior sólo nos interesa cuando estamos hablando de tiempos de muestreo de segundos):
$$
\begin{equation} \tag{1}
\rho·c_p·A_{cs}·\frac{\delta T_{out}(t)}{\delta t}=\beta I(d_I) - \frac{H}{L_{eq}}(\tilde{T}(t)-T_{amb}(t)) - \frac{c_{p}}{c_{f}}· \frac{1}{L_{eq}}·\dot{m}_{sf}(t-d*\dot{m})·(T_{out}(t)-T_{in}(t-d_{Tin}))
\end{equation}
$$
$$
\begin{equation} \tag{2}
\tilde{T}(t) = \frac{T_{out}(t)+T_{in}(t-d_{Tin})}{2}
\end{equation}
$$
$$ L_{eq}=n_s·L_t $$
$$ c_f=n_p·n_t $$

Desarrollando en diferencias, $\frac{\delta T_{out}(t)}{\delta t}$ pasa a ser $\frac{T_{out}(k)-T_{out}(k-1)}{\Delta T}$, retardos no se consideran y despejando $\dot{m_{sf}}$:
$$
\begin{equation} \tag{3}
\dot{m}_{sf} =\frac{ -\rho·c_{p,\tilde{T}}·A_{cs}·\frac{T_{out}(k)-T_{out}(k-1)}{\Delta T} - \beta I(k) + \frac{H}{L_{eq}}(\tilde{T}-T_{amb}(k)) }{  \frac{c_{f}}{c_{p,\tilde{T}}}· L_{eq}·(T_{out}(k)-T_{in}(k)) }
\end{equation}
$$
$$
\begin{equation} \tag{4}
\tilde{T}(k) = \frac{T_{out}(k)+T_{in}(k)}{2}
\end{equation}
$$

**Antiguo:**
En estacionario $T_{out}$ no varía, y los retardos no importan, desarrollando la expresión y despejando $T_{out}$ resulta:
$$
\begin{equation} \tag{3}
T_{out}=I·\frac{\beta}{\frac{1}{L_{eq}}\left( \frac{H}{2}+ \frac{c_{p}}{c_{f}}·\dot{m}_{sf} \right)} + T_{in}·\frac{\dot{m}_{sf}- \frac{H·c_{f}}{2·c_{p}}}{\dot{m}_{sf}+ \frac{H·c_{f}}{2·c_{p}}} + T_{amb}·\frac{2}{1+ \frac{2·c_{p}}{c_{f}·H}·\dot{m}_{sf}}
\end{equation}
$$
$$
\begin{equation} \tag{4}
P_{gen}=\dot{m}_{sf}·c_{p}·(T_{out}-T_{in}) \: [kW_{th}]
\end{equation}
$$

Tendría que haber despejado $\dot{m}_{sf}$ no $T_{out}$ 🤕

## Implementation

Implemented in `models_psa.solar_field`.

## Otros

![Campo solar](Validation%20signals%20list.md#Campo%20solar)