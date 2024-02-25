# Multi-effect distillation plant model
## Nomenclature
![](../attachments/Diagrama%20general.svg)
- Hot water
	- $\dot{m}_s$  $[m^3/h]$ Hot water flow rate
	- $T_{s,in}$  $[\degree C]$ Hot water inlet temperature
	- $T_{s,out}$  $[\degree C]$ Hot water outlet temperature
- Feed water
	- $\dot{m}_f$  $[m^3/h]$ Feed water flow rate
	- $T_f$  $[\degree C]$ Feed water temperature at first effect (preheated seawater)
- Cooling water
	- $\dot{m}_{c}$  $[m^3/h]$ Cooling water flow rate
	- $T_{c,in}$  $[\degree C]$ Cooling water temperature at inlet of condenser
	- $T_{c,out}$  $[\degree C]$ Cooling water temperature at outlet of condenser
- Products
	- Condensate
		- $\dot{m}_{d}$  $[m^3/h]$ Distillate production flow rate
		- $T_d$  $[\degree C]$ Distillate production temperature
		- $\omega_d$  $[g/L]$ Distillate production salinity
	- Brine
		- $\dot{m}_{b}$  $[m^3/h]$ Brine flow rate
		- $T_{b}$  $[\degree C]$ Brine temperature
		- $\omega_{b}$  $[g/L]$ Brine salinity
- Others
	- $T_{v,1}$  $[\degree C]$ Vapor temperature in first effect
	- $T_{v,N}$  $[\degree C]$ Vapor temperature in last effect
	- $T_{v,c}$  $[\degree C]$ Vapor temperature in condenser
	- $t_{operated,i}$  $[hours]$ Time that the i th-effect has been operated since last cleanup

## Fuente


## Inputs / outputs

$$ \dot{m}_{d}, T_{s,out}, \dot{m}_{c} = f(\dot{m}_{s}, T_{s,in}, \dot{m}_{f}, T_{c,in}, T_{c,out}, [t_{operated, i}]_{i=1..N_{ef}}) $$
$$ \dot{m}_{d}, T_{s,out}, \dot{m}_{c} = f(\dot{m}_{s}, T_{s,in}, \dot{m}_{f}, T_{c,in}, T_{c,out}) $$

Reasoning behind chosen outputs:
- $\dot{m}_{d}$ is part of the cost function.
- $T_{s,out}$. Necessary to estimate thermal performance.
- $\dot{m}_{c}$. Necessary to estimate electrical performance and feasibility of operation.


## Análisis sensibilidad

En esta parte me he atascado un poco, no tengo claro de si se necesita evaluar en qué magnitud variaciones en cada una de las entradas afectan a la salida o el peso/importancia relativa de cada una de las entradas en el modelo. De cualquier manera ambas cosas me parece interesante estudiarlas.

Para lo primero se me ocurren dos aproximaciones, la primera sería evaluar el modelo en todo el espacio de estados, he leído que se puede evaluar con simulaciones de [monte carlo](https://es.mathworks.com/help/sldo/ug/what-is-sensitivity-analysis.html), pero no he llegado a probar. Otra opción sería analizar directamente los datos en bruto, para ello he probado por un lado a realizar un análisis de varianza [ANOVA](https://es.mathworks.com/help/stats/n-way-anova.html) y por otro lado [PCA](https://www.wikiwand.com/en/Principal_component_analysis).

Para el ANOVA este es el resultado:
![](Pasted%20image%2020221011134640.png)
Si para que un factor (entrada) tenga significancia en la salida ($M_{prod}$) su p-value debe ser inferior a 0.05 ([Output arguments -- p-values](https://es.mathworks.com/help/stats/anovan.html)) entonces los factores relevantes son:

|     | Ms       | Ts_in    | Mf       | Tcwin    | Tcwout   | Ms*Ts_in | Ms*Mf    | Ms*Tcwin | Ms*Tcwout | Ts_in*Mf | Ts_in*Tcwin | Ts_in*Tcwout | Mf*Tcwin | Mf*Tcwout | Tcwin*Tcwout |
| --- |----------|----------|----------|----------|----------|----------|----------|----------|-----------|----------|-------------|--------------|----------|-----------|--------------|
|  **p.values**   | 1.57E-03 | 3.33E-39 | 1.41E-01 | 1.90E-13 | 2.10E-39 | 4.67E-01 | 2.93E-01 | 8.65E-01 | 6.95E-01  | 3.94E-01 | 2.02E-07    | 4.58E-05     | 9.51E-01 | 4.35E-01  | 3.15E-06     |
|  **p<0.05**     | 1        | 1        | 0        | 1        | 1        | 0        | 0        | 0        | 0         | 0        | 1           | 1            | 0        | 0         | 1            |


En cuanto al PCA, entiendo que esto es algo que normalmente se hace para intentar reducir la cantidad/complejidad de entradas/parámetros que requiere un modelo a costa de perder el sentido físico de las variables. Pero he visto que puede ser útil para analizar la relación entre variables con visualizaciones tipo [biplot](https://es.mathworks.com/help/stats/biplot.html). Esta sería para el sistema de estudio:
![](biplot.jpeg)
Según esto interpreto que $M_{prod}$ es dependiente/está relacionado con $M_{cw}$, $T_{f}$, $T_{s,in}$ y $T_{s,out}$ por estar próximos en el eje del componente principal 1 y también $M_s$ y $M_f$ por estarlo en el del segundo. No así para $T_{cw,in}$ y $T_{cw,out}$, lo cual me extraña ya que la operación del condensador es fundamental para el funcionamiento de la planta. Se puede deber a que aquí sólo se muestran dos componentes principales de todos los disponibles. No sé si veis que esta aproximación tenga sentido.

Para lo segundo (peso relativo) he encontrado:
![1 Ecuación usada para estudio sensibilidad de salida a las distintas entradas|500](Pasted%20image%2020221010153819.png)
[1]. Ecuación usada para estudio sensibilidad de salida a las distintas entradas

donde $E_j$, $W$ y $N$ son la importancia relativa de la j-ésima variable de entrada en el coeficiente ponderado y el número de neuronas, respectivamente. Los subíndices $k$, $t$ y $n$ se refieren a las neuronas de entrada, ocultas y de salida, respectivamente, y los superíndices $i$, $h$ y $o$ se refieren a las capas de entrada, ocultas y de salida, respectivamente ([fuente](https://es.mathworks.com/matlabcentral/answers/299646-how-to-obtain-the-relative-importance-of-each-input-variable-for-a-neural-network#answer_783049)).

##### Opción 1. Red que provee de $M_{prod}$, $T_{s,out}$ y $M_{cw}$
Importancia relativa (%) de cada una de las entradas para cada una de las salidas del modelo usando [1]:

|      | Ms   | Ts_in | Mf   | Tcwout | Tcwin | op_time_1 | preprecalentador |
| ---- | ---- | ----- | ---- | ------ | ----- | --------- | ---------------- |
|  $M_{prod}$    | 23.4 | 13.1  | 11.7 | 13.2   | 15.4  | 9.4       | 13.9             |
|  $T_{s,out}$    | 14.7 | 6.6   | 15.4 | 19.0   | 17.2  | 14.1      | 13.0             |
|  $M_{cw}$    | 16.5 | 9.5   | 15.7 | 13.3   | 12.7  | 21.5      | 10.7             |

Si los datos obtenidos de [1] son representativos de la red parece que esta opción no es válida a pesar de dar errores bajos pues no se le da peso a entradas importantes en función de la salida a evaluar ($T_{s,in}$ -> $T_{s,out}$) o demasiado peso a entradas con poca relevancia física para una salida ($t_{operated,[1]}$->$M_{cw}$).

##### Opción 2. Redes individuales
- Red $M_{prod}$, entradas: $M_{s}$, $T_{s,in}$, $M_{f}$, $t_{operated,[1:3]}$  
Importancia relativa (%) de cada una de las entradas para cada una de las salidas del modelo usando [1]:

| Ms   | Ts_in | Mf   | Tcwout | op_time_1 |
| ---- | ----- | ---- | ------ | --------- |
| 25.6 | 23.1  | 10.2 | 20.6   | 20.5      |

- Red $T_{s,out}$, entradas: $M_{s}$, $T_{s,in}$, $M_{f}$, $T_{cw,out}$, $t_{operated,[1]}$
Importancia relativa (%) de cada una de las entradas para cada una de las salidas del modelo usando [1]:

| Ms               | Ts_in | Mf              | Tcwout | op_time_1 |
|------------------|-------|-----------------|--------|-----------|
| 33.7 | 17.4  | 11.6 | 11.4   | 25.8      |

- Red $M_{cw}$, entradas: $M_{s}$, $T_{s,in}$, $M_{f}$, $T_{cw,in}$, $T_{cw,out}$
Importancia relativa (%) de cada una de las entradas para cada una de las salidas del modelo usando [1]:

| Ms   | Ts_in | Mf   | Tcwout | Tcwin | preprecalentador |
|------|-------|------|--------|-------|------------------|
| 21.0 | 10.6  | 17.4 | 17.1   | 15.6  | 18.3             |

En algunos ensayos una válvula manual que no debería estar abierta hacía que parte del agua tras abandonar el condensador circulase por un tramo de tubería mayor y se enfriara, esto lo identifico con la variable lógica preprecalentador. Probablemente sea mejor directamente eliminar esos datos para esta red. En las otras dos salidas no supone un problema porque se usa $T_{cw,out}$ como entrada.

### Consumo térmico específico (STEC):
Con $M_{prod}$ y $T_{s,out}$:
$$STEC = \frac{M_{s}·c_p·(T_{s,in}-T_{s,out})}{M_{prod}} · \rho_{prod}· \frac{1\:kWh}{3600\:kJ} \: \left[ \frac{kWh_{th}}{m^{3}} \right]$$
### Consumo eléctrico específico (SEEC):
- Bomba destilado ($M_{prod}$) -> Variador con medidas de consumo eléctrico
- Bomba salmuera ($M_{brine}$) -> Variador con medidas de consumo eléctrico
- Bomba alimentación ($M_{f}$) -> Medida junto con $M_s$, ajustar curva
- Bomba agua de mar ($M_{cw}$) -> Variador con medidas de consumo eléctrico
- ~~Bomba vacío~~ -> Medida junto con $M_{prod}$ y $M_{brine}$, se puede calcular (no considerar en principio)
- Bomba agua caliente ($M_s$) -> Medida junto con $M_f$, ajustar curva
$$SEEC= \frac{P_{M_{prod}} + P_{M_{brine}} + P_{M_{f}} + P_{M_{cw}} + P_{M_{s}}}{M_{prod}}·t \left[ \frac{kWh_{e}}{m^{3}} \right]$$
