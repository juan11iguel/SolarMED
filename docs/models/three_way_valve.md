
# Three-way valve model

**NOTA:** Puede ser que a veces aparezcan nombres de variables sin o con el prefijo del sistema ($3wv$), pero son iguales.

## Inputs / outputs

$$ \dot{m}_{src}, R = f(\dot{m}_{dis}, T_{src}, T_{dis,in}, T_{dis,out}) $$
## Fuente

Lidia

## Nomenclature

![center](attachments/solarMED_optimization-Three-way%20valve%20model.png)

- $T_{src} \: (\degree C):$ Temperature from source (thermal storage)
- $T_{dis,in} \: (\degree C):$ Inlet temperature to discharge (MED heat source inlet, $T_{med,s,in}$)
- $T_{dis,out} \: (\degree C):$ Outlet temperature from discharge (MED heat source outlet, $T_{med,s,out}$)
- $\dot{m}_{dis} \: (kg/s):$  Flow rate through load / discharge
- $\dot{m}_{src} \: (kg/s)=(1-R)·\dot{m}_{dis} :$  Flow rate from source
- $\dot{m}_{mix}=R·\dot{m}_{dis} \: (kg/s):$  Fraction of flow rate from discharge that is mixed with source
- $R$: Ratio of $\dot{m}_{dis}$ that is mixed with $\dot{m}_{src}$


## Equations

Balance de materia:

$$
\begin{equation} \tag{1}
\dot{m}_{mix}=\dot{m}_{dis}-\dot{m}_{src}
\end{equation}
$$
$$ \dot{m}_{src}=(1-R)\:\dot{m}_{dis} $$
Balance de energía:
$$
\begin{equation} \tag{2}
\dot{m}_{src}·T_{src}·c_p+\dot{m}_{mix}·T_{dis,out}·c_p=\dot{m}_{dis}·T_{dis,in}·c_{p}
\end{equation}
$$

Sustituyendo (1) en (2) y despejando:
$$
\begin{equation} \tag{3}
\dot{m}_{src} = \dot{m}_{dis}·\frac{T_{dis,in}·c_{p,dis,in}-T_{dis,out}·c_{p,dis,out}}{T_{src}·c_{p,src}-T_{dis,out}·c_{p,dis,out}}
\end{equation}
$$
También se puede obtener directamente R a partir de temperaturas:
$$\begin{equation} \tag{4}
R = \dot{m}_{dis}·\frac{T_{dis,in}·c_{p,dis,in}-T_{src}·c_{p,src}}{T_{dis,out}·c_{p,dis,out}-T_{src}·c_{p,src}}
\end{equation}
$$

# Resultados

Resultados reproducibles ejecutando [test_models.py](../models.psa/test_models.py) - `Test three-way valve` para evaluar modelo y representación de resultados.


Esta es la prueba comparando el uso de distintas señales como entradas a la función:

![](Figure_3wv_model.svg)

Ejecutando el modelo para varios días donde se ha operado la planta estos son los resultados:

![result_model_3wv_2023-04-14](result_model_3wv_2023-04-14.svg)
![result_model_3wv_2023-06-21](result_model_3wv_2023-06-21.svg)

## Otros

![Válvula de tres vías](Validation%20signals%20list.md#Válvula%20de%20tres%20vías)