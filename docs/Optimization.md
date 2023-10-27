# Optimización

## Caso 1. MED

Sólo la planta MED, la optimización de este caso se ha estudiado ampliamente en la literatura. Normalmente las plantas se diseñan para funcionar en unas condiciones nominales donde obtienen el máximo rendimiento, fuera de ahí los intercambiadores de calor no funcionan en el punto óptimo (diferencia de temperatura entre entrada y salida fuera de diseño). Por lo que el resultado de este caso debería ser similar a las condiciones nominales de la planta y no debería cambiar mucho con el tiempo. Es un sistema que evoluciona poco con el tiempo a excepción de la pérdida de rendimiento en la transferencia de calor en los intercambiadores debido a ensuciamiento/encrustación de sales y el aumento de la temperatura del agua de entrada en el condensador (particular a planta PSA).

![[MED_case1.svg]]
$$\min\limits_{U}  (a·STEC + b·SEEC - M_{prod}·c·T_s)$$
s.t.:
- $25.2 \le M_{s} \le 50.4$ $(m^3/h)$
- $5 \le M_{f} \le 9$ $(m^3/h)$
- $10 \le M_{cw} \le 24$ $(m^3/h)$
- $1 \le M_{prod} \le 3$ $(m^3/h)$
- $62 \le T_{s,in} \le 74$ $(\degree C)$
- $25 \le T_{v,c} \le 35$ $(\degree C)$
- $11.5 \le T_{cw,in} \le 34$ $(\degree C)$
- $15 \le T_{cw,out} \le 40$ $(\degree C)$

donde:
- $U = [M_{s}, M_{f}, T_{cw,out}, T_{s,in}]$
- $a$ representa el precio unitario de energía térmica ($\texteuro/kWh_{th}$). Si se considera proveedor externo, asignar coste. Si es producción propia, habría que considerar SEEC campo solar, costes de inversión etc (más detallado en [Caso 3 MED Almacenamiento térmico Campo solar](#Caso%203%20MED%20Almacenamiento%20térmico%20Campo%20solar))
- $b$ precio unitario de energía eléctrica ($\texteuro/kWh_e$). Habrá que aplicar corrección o bien a $SEEC_{MED}$ o bien a $b$ para corregir consumo alto de nuestra planta y considerar consumo de planta típico + variaciones medidas según punto de operación en la nuestra.
- $c$ precio unitario de venta de agua ($\texteuro/m^3$)
- $STEC$. Consumo energía térmica específico ($kWh_{th}/m^3$)
- $SEEC$. Consumo de energía eléctrica específico ($kWh_{e}/m^3$)
- $M_{prod}$ Caudal de destilado producido ($m^3/h$)


### Diagrama del sistema simulado
Sistema simulado usado para entrenar agente de RL. 

![Simulated_system_diagram](attachments/Simulated_system_diagram.svg)

### Arquitectura sistema óptimo. RL en capa optimización

![optimal_system_diagram_RL](attachments/optimal_system_diagram_RL.svg)



<font style="color:red">A partir de aquí no está terminado/revisado...</font>

## Caso 2. MED + Almacenamiento térmico

Se parte de un fluido con una energía finita a una temperatura determinada. La idea es gestionar esta energía disponible de manera que se maximice la producción total de destilado. Este caso es más interesante pues operar en condiciones nominales puede no ser la mejor estrategia. En este caso el factor tiempo es fundamental. Se pueden plantear distintos puntos de partida y distintas capacidades de almacenamiento térmico. Como referencia el almacenamiento instalado para la MED de la PSA proporciona autonomía para unas 2-5 h dependiendo de las condiciones de operación.

![MED_case2](attachments/MED_case2.svg)


$$\min\limits_{U} \left(a·STEC + b·SEEC - M_{prod}·c·t)
\right|_{t=t_0}^{t=\infty}$$
- $25.2 \le M_{s} \le 43.2$ $(m^3/h)$
- $6 \le M_{f} \le 10$ $(m^3/h)$
- $M_{prod}, T_{s,out} = f(M_{s}, T_{s,in}, M_{f}, M_{cw}, T_{cw,in}, [t_{operated, i}]_{i=1..N_{ef}})$
- $T_{tank} \ge T_{tank,min}$
- etc

Se parte de un volumen en los tanques a una temperatura determinada ($T_{tank,0}$) y el sistema evoluciona hasta un estado de equilibrio donde no se puede producir más agua ($T_{tank,\infty}$). El objetivo es maximizar el volumen de agua producido (mientras sea rentable producir) con la energía finita disponible.


## Case study 3. MED + thermal storage + solar field

(sustituir diagrama)
![Hierarchical Control](attachments/Hierarchical%20Control.svg)

Pendiente de actualizar nomenclatura:
![solarMED_optimization-general_diagram](solarMED_optimization-general_diagram.svg)

Pendiente de actualizar nomenclatura:
![center | solarMED_optimization-Optimization.drawio | 700](models/attachments/solarMED_optimization-Optimization.drawio.svg)

### Cost function
$$J = \max\limits_{U} \left( (C_w-C_{op})·\dot{m}_{med,d} - C_{fixed} \right)·t_{s} \: \left[ u.m. \right]$$
where:
- $C_{op} \: \left( \frac{\texteuro}{m^3·h} \right) = C_{e}\left[\frac{\texteuro}{kWh_e}\right]·(SEEC_{med}\left[\frac{kW_{e}}{m^3}\right]+STEC_{med}\left[\frac{kW_{th}}{m^3}\right]·SEC_{sf}\left[\frac{kW_{e}}{kW_{th}}\right])$
- $C_{fixed} \: \left( \frac{\texteuro}{h} \right) = C_{MED}+C_{e.storage}+C_{e.generation}$
- $C_{w} \: \left( \texteuro/m^3 \right)$ Sale price of water
- $C_{e} \: \left( \texteuro/kWh_e \right)$. Cost of the electrical energy, it should depend on the time. Could be taken directly from some database.
- $t_{s}$ Sample time between optimizations

### Decision variables

Optimizer controllable inputs: $U = (U_{sf}, U_{med}, U_{ts})$
- Water production $(U_{med})$
	- $T_{med,s,in}$
	- $\dot{m}_{med,s}$
	- $\dot{m}_{med,f}$
	- $T_{c,out}$
- Energy production $(U_{sf})$
	- $T_{sf,out}$
- Energy storage $(U_{ts})$
	- $\dot{m}_{ts,src}$

###  Environment variables

- Ambient temperature $(T_{amb})$ 
- Solar radiation $(I)$
- Seawater temperature $(T_{c,in})$
- Seawater salinity $(w_f)$ 

This is assuming the heat is provided by a self-owned facility. If the heat is provided by an external provider, then the fixed costs should not include the costs associated with the energy generation and the operational costs are obtained as: $C_{op}=C_e·SEEC_{MED} + C_{th}·STEC_{MED}$ where $C_{th} \: (\texteuro/kWh_{th})$ is the sale price of the heat, it should be a function that increases its output based on the temperature at what the heat is needed, or as with the electricity cost depend on the time as well.

### Observaciones

Sacar todo esto va a tener trabajo, podemos empezar intentando ver valores típicos o usar análisis económicos que haya hechos en la literatura (Patri has hecho alguno para MED no?), luego irá que ir jugando con los pesos porque la instalación que estamos evaluando no tiene demasiado sentido económico (quitarle peso a los costes fijos). También una vez tengamos todo hecho, jugar con los pesos para ver cómo afecta a las decisiones.

### Restricciones

Restrictions. (WIP)
s.t.:
- $25.2 \le M_{s} \le 43.2$ $(m^3/h)$
- $6 \le M_{f} \le 10$ $(m^3/h)$
- $T_{tank} \ge T_{tank,min}$


~~NOTA: Me he dado cuenta de que he puesto mucho generación, igual tendría que haber usado producción~~

### Diagrama sistema simulado

![center | solarMED_optimization-Complete system model.drawio](solarMED_optimization-Complete%20system%20model.drawio.svg)


Implementado en `models_psa.solarMED` (WIP).