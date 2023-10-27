
**NOTA:** Puede ser que a veces aparezcan nombres de variables sin el prefijo del sistema ($ts$), pero son iguales.

Es un sistema de 2 tanques que garantiza que haya estratificación, uno de los tanques se encuentra a mayor temperatura (tanque rojo en diagrama), recibe el calor de la fuente de energía y cede calor al consumo. La descarga del consumo retorna a la parte inferior del tanque frío (azul en diagrama) antes de circular hacia el tanque caliente absorbiendo calor de la fuente. 

%% En estacionario tiene que cumplirse que la energía que se aporta al almacenamiento es la misma que se consume y hay un equilibrio de temperaturas. Se puede simplificar a un único tanque con un volumen la suma de ambos. %% 

![center | solarMED_optimization-Storage model.drawio](solarMED_optimization-Storage%20model.drawio.svg)
## Inputs / outputs

$$ T_{i},E_{avail} = f(T_{T,in},T_{B,in},\dot{m}_{src},\dot{m}_{dis},T_{i,k-1},t_{s},T_{min},T_{amb},(UA)_{i}, V_i) $$
$$  T_{h}, T_{c} = f(T_{T,in},T_{B,in},\dot{m}_{src},\dot{m}_{dis},T_{h,k-1}, T_{c,k-1},t_{s},T_{amb},(UA)_{i}, V_i)  $$

La estratificación significa que en el tanque coexisten i volúmenes de fluido a distintas temperaturas (en realidad es un gradiente de temperaturas), definiendo dos volúmenes ($V_T$, $V_B$), la energía útil almacenada será  $E_{net}=E_T+E_B$ (si $T_{min}>T_{B} \rightarrow E_B=0$): 
$$ E_{avail}=E_T+E_B=\rho_T·V_T·c_{p,Tavg}·\Delta T_{T,net}+\rho_B·V_B·c_{p,Bavg}·\Delta T_{B,net}=\rho_T·V_T·c_{p,Tavg}·(T_T-T_{min})+\rho_B·V_B·c_{p,Bavg}·(T_B-T_{min}) $$

Para N volúmenes sería $\sum\limits E_{i} \: if \:T_i>T_{min}$ 


## Fuente

Lidia y genialidad propia

### Nomenclatura

- $T_{T} \: (\degree C):$ Temperature at the top half of the tank
- $T_{B} \: (\degree C):$ Temperature at the bottom half of the tank
- $T_{B,in} \: (\degree C):$ Inlet temperature to bottom of the tank after load
- $T_{B,out}=T_B \: (\degree C):$ Outlet temperature from bottom of the tank to heat source
- $T_{T,in} \: (\degree C):$  Inlet temperature from heat source to top of the tank
- $T_{T,out}=T_T \: (\degree C):$ Outlet temperature from top of the tank to load
- $\dot{m}_{dis}\: (kg/s):$  Flow rate of energy sink (consumer)
- $\dot{m}_{src}\: (kg/s):$  Flow rate of heat source
- $E_{avail} \: (kWh_{th}):$ Useful thermal energy stored in tank system
- $T_{min} \: (\degree C):$ Useful temperature limit
- $T_{amb} \: (\degree C):$ Ambient temperature

### Señales SCADA

| Name             | ID               | Units    |
| ------------------ | ---------------- | :-----------: |
| $T_{ts,h,t}$       | TT-SF-004        | $^{\circ}C$ |
| $T_{ts,h,m}$       | TT-SF-005        | $^{\circ}C$ |
| $T_{ts,h,b}$       | TT-SF-006        | $^{\circ}C$ |
| $T_{ts,c,t}$       | TT-SF-001        | $^{\circ}C$ |
| $T_{ts,c,m}$       | TT-SF-002        | $^{\circ}C$ |
| $T_{ts,c,b}$       | TT-SF-003        | $^{\circ}C$ |
| $T_{amb}$          | TT-DES-030       | $^{\circ}C$ |
| $\dot{m}_{ts,src}$ | FT-SF-001        | L/min       | 
| $\dot{m}_{ts,dis}$ | FT-AQU-100 o 101 (comprobar si mide cuando 100 activa) | L/s            |
| $T_{ts,t,in}$      | TT-SF-008        | $^{\circ}C$ |
| $T_{ts,b,in}$      | TT-AQU-109       | $^{\circ}C$ |

## Equations

![center | 500](modelo_tanques%20-%20Selection%201.png)
![center  | 400](Pasted%20image%2020230703135102.png)


### Volumen interior
$$
\begin{equation} \tag{1}
-\rho·V_{i}·c_p·\frac{T_{i,k}-T_{i,k-1}}{t_{s}}+ \dot{m}_{src}·T_{i-1,k}·c_{p}-\dot{m}_{dis}·T_{i,k}·c_p-\dot{m}_{src}·T_{i,k}·c_p+\dot{m}_{dis}·T_{i+1,k}·c_p-UA_{i}·(T_{i,k}-T_{amb}) = 0
\end{equation}
$$

### Volumen superior

$$
\begin{equation} \tag{2}
-\rho·V_{T}·c_p·\frac{T_{T,k}-T_{T,k-1}}{t_{s}}+ \dot{m}_{src}·T_{T,in}·c_{p}-\dot{m}_{dis}·T_{T,k}·c_p-\dot{m}_{src}·T_{T,k}·c_p+\dot{m}_{dis}·T_{1,k}·c_p-UA_{T}·(T_{T,k}-T_{amb}) = 0
\end{equation}
$$

### Volumen inferior

$$
\begin{equation} \tag{3}
-\rho·V_{B}·c_p·\frac{T_{B,k}-T_{B,k-1}}{t_s}+\dot{m}_{src}·T_{i-1,k}·c_{p}+\dot{m}_{dis}·T_{B,in}·c_p-\dot{m}_{src}·T_{B,k}·c_p-\dot{m}_{dis}·T_{B,k}·c_p-UA_{N}·(T_{B,k}-T_{amb}) = 0
\end{equation}
$$

%%
## Equations

$$
\begin{equation} \tag{1}
\rho·V_{T}·c_p·\frac{T_T-T_{T,k-1}}{t_s}= \dot{m}_{src}·T_{T,in}·c_{p}-\dot{m}_{dis}·T_{T,k-1}·c_p-\dot{m}_{src}·T_{T,k-1}·c_p+\dot{m}_{dis}·T_{B}·c_p-UA·(T_T-T_{amb})
\end{equation}
$$
$$
\begin{equation} \tag{2}
\rho·V_{B}·c_p·\frac{T_B-T_{B,k-1}}{t_s}= \dot{m}_{src}·T_{T}·c_{p}+\dot{m}_{dis}·T_{B,in}·c_p-\dot{m}_{src}·T_B·c_p-\dot{m}_{dis}·T_{B}·c_p-UA·(T_B-T_{amb})
\end{equation}
$$

Despejando $T_T$ de (1) y aproximando que los flujos se calculan con $T_{k-1}$ (esto hay que comprobar no introduzca error comparando con modelo dinámico):
$$
\begin{equation} \tag{3}
T_T=T_{T,k-1} + t_s· \frac{\dot{m}_{src}·T_{T,in}·c_{p,Tin}-\dot{m}_{dis}·T_{T,k-1}·c_{p,T_{k-1}}-\dot{m}_{src}·T_{T,k-1}·c_{p,T_{k-1}}+\dot{m}_{dis}·T_{B,k-1}·c_{p,B_{k-1}}-UA·(T_{T,k-1}-T_{amb})}{\rho·V_{T}·c_{p,T_{k-1}}}
\end{equation}
$$
$$
\begin{equation} \tag{4}
T_B=T_{B,k-1} + t_s· \frac{\dot{m}_{src}·T_{T,k-1}·c_{p}+\dot{m}_{dis}·T_{B,in}·c_{p,Bin}-\dot{m}_{src}·T_{B,k-1}·c_{p,B_{k-1}}-\dot{m}_{dis}·T_{B,k-1}·c_{p,B_{k-1}}-UA·(T_{B,k-1}-T_{amb})}{\rho·V_{B}·c_{p,B_{k-1}}}
\end{equation}
$$

# Aproximación cuando hay desequilibrio Pin≠Pout
# Aproximación cuando Pin=Pout

### Inputs / outputs

$$ T_{T},T_{B},E_{net} = f(T_{T,in},T_{B,in},\dot{m}_{src},\dot{m}_{dis},T_{min},T_{amb},(UA)_{tk}) $$
$UA$ is a parameter to be calibrated. It depends on the heat exchange surface, which is the total outer surface of the tanks exposed to the environment (if known it can just be substituted) and the heat transfer coefficient, which depends on the temperature difference between the ambient and temperature of the volume.

## Equations

Puesto que $\rho·V_{B/T}·c_p·\frac{dT_{T/B}}{dt}=0$:

$$
\begin{equation} \tag{1}
\dot{m}_{src}·T_{T,in}·c_{p}-\dot{m}_{dis}·T_T·c_p-\dot{m}_{src}·T_T·c_p+\dot{m}_{dis}·T_{B}·c_p-UA·(T_T-T_{amb})=0
\end{equation}
$$
$$
\begin{equation} \tag{2}
\dot{m}_{src}·T_{T}·c_{p}+\dot{m}_{dis}·T_{B,in}·c_p-\dot{m}_{src}·T_B·c_p-\dot{m}_{dis}·T_{B}·c_p-UA·(T_B-T_{amb})=0
\end{equation}
$$
System of two equations with two unknown variables ($T_T,T_B$), despejando por ejemplo $T_B$ de ambas:

De (1):
$$
\begin{equation} \tag{3}
T_{B}=\frac{ UA·(T_T-T_{amb})+T_T·(\dot{m}_{dis}+\dot{m}_{src})·c_{p,T}-T_{T,in}·\dot{m}_{src}·c_{p,T,in} }{\dot{m}_{dis}·c_{p,B}}
\end{equation}
$$
De (2):
$$
\begin{equation} \tag{4}
T_{B}=\frac{ T_T·\dot{m}_{src}·c_{p,T}+T_{B,in}·\dot{m}_{dis}·c_{p,B,in}+UA·T_{amb} }{UA+(\dot{m}_{src}+\dot{m}_{dis})·c_{p,B}}
\end{equation}
$$
Igualando (3) y (4) y despejando $T_T$, cinco chorizos después resulta:
$$
\begin{equation} \tag{5}
T_{T}=\frac{ T_{amb}·( (UA)^2+UA·c_{p,B}·(\dot{m}_{src}+2·\dot{m}_{dis})+ T_{T,in}·( \dot{m}_{src}·c_{p,T,in}·(UA+c_{p,B}(\dot{m}_{src}+\dot{m}_{dis}))  )+ T_{B,in}·(\dot{m}_{dis}^2·c_{p,B}·c_{p,B,in})}{ (UA)^2+UA·(\dot{m}_{src}+\dot{m}_{dis})(c_{p,B}+c_{p,T})+ (\dot{m}_{src}+\dot{m}_{dis})^2·c_{p,T}·c_{p,B}-\dot{m}_{src}·\dot{m}_{dis}·c_{p,T}·c_{p,B}}
\end{equation}
$$
Con $T_T$ despejar en cualquiera de (1) o (2) para obtener $T_B$.

Si alguien es tan amable de comprobar los cálculos 🙄😅
%%


# Resultados


## Resultados V2

En esta segunda versión se modelan los tanques individualmente incluyendo la temperatura de la parte de abajo y se resuelven secuencialmente, primero uno y con el resultado de ese se resuelve el segundo. El orden depende de los valores de $\dot{m}_{src}$, $\dot{m}_{dis}$.

Resultados reproducibles ejecutando:
- [calibrate_model_thermal_storage2.py](../models.psa/calibrate_model_thermal_storage2.py) para calibración de parámetros de modelo
- [test_models.py](../models.psa/test_models.py) - `Test thermal storage model V2` para evaluar modelo y representación de resultados finales (ajuste de temperatura y energía almancenada)

1. Datos de partida. Línea azul del eje y derecho representa la evolución de la temperatura ambiente. Se representa un día donde se realiza tanto la carga del tanque ($\dot{m}_{src}\gt 0$) como la descarga ($\dot{m}_{dis}\gt 0$). Puesto que el caudalímetro que da la señal ($\dot{m}_{dis}$) no funciona se estima mediante modelo de válvula de tres vías.
   ![](Figure_1.png)
   
2. Primera prueba, usando los parámetros de V1 por probar a ver si el modelo tiene sentido, más o menos parece que sí, así que se puede continuar a intentar ajustar parámetros.
![](Figure_8.png)

4. Prueba de calibración con mismos datos que V1:
   ![result_model_ts_calibration_2023-06-02](result_model_ts_calibration_2023-06-02.svg)

Parámetros:

| Variable | $T_{ts,h,t}$ | $T_{ts,h,m}$ | $T_{ts,h,b}$ | $T_{ts,c,t}$ | $T_{ts,c,m}$ | $T_{ts,c,b}$ |
|----------|--------------|--------------|--------------|--------------|--------------|--------------|
| UA       | 5.61e-03     | 2.26e-03     | 4.77e-02     | 1.02e-02     | 2.99e-03     | 1.13e-01     |
| V        | 2.45e+00     | 4.86e+00     | 2.41e+00     | 4.51e+00     | 1.34e+00     | 1.00e+01     |

Evaluando desempeño con otros datos:

![result_model_ts_V2_2023-04-14](result_model_ts_V2_2023-04-14.svg)
![result_model_ts_V2_2023-06-21](result_model_ts_V2_2023-06-21.svg)

5. Tras cambios en instrumentación, se hace un ensayo con datos que incluyen sólo carga, sólo descarga, carga y descarga simultánea y período sin carga ni descarga. Dos días ejecutándose después:
![result_model_ts_calibration_2023-07-07](result_model_ts_calibration_2023-07-07.svg)

| Variable | $T_{ts,t}$ | $T_{ts,m}$ | $T_{ts,b}$ |
| -------- | ---------- | ---------- | ---------- |
| $UA_h$   | 0.0069818  | 0.00584034 | 0.03041486 |
| $V_h$    | 5.94771006 | 4.87661781 | 2.19737023 |
| $UA_c$   | 0.01396848 | 0.0001     | 0.02286885 |
| $V_c$    | 5.33410037 | 7.56470594 | 0.90547187 |


## Resultados V1

Resultados reproducibles ejecutando:
- [calibrate_model_thermal_storage.py](../models.psa/calibrate_model_thermal_storage.py) para calibración de parámetros de modelo
- [test_models.py](../models.psa/test_models.py) - `Test thermal storage model` para evaluar modelo y representación de resultados finales (ajuste de temperatura y energía almancenada)

1. Datos de partida (quitando volúmenes inferiores de cada tanque), línea azul del eje y derecho representa la evolución de la temperatura ambiente. Se representan tres días donde en tres períodos se realiza la carga del tanque ($\dot{m}_{src}\gt 0$).
   ![center](Pasted%20image%2020230612161118.png)
2. Primera prueba ajustando a ojo coeficiente de pérdidas con el ambiente (UA, igual habría sido mejor llamarlo H). Se toma una ventana temporal donde no se carga el tanque y sólo hay disminución de temperatura por intercambio con el ambiente.
   ![center](Pasted%20image%2020230612161342.png)
   3. Tras realizar el proceso de calibración, se obtienen los coeficientes de pérdidas ajustados y un buen ajuste de temperatura cuando las pérdidas con el ambiente es el único fenómeno a considerar:
      ![](Pasted%20image%2020230612192550.png)
   4. Aún así se puede observar que la respuesta no es buena cuando se considera carga / descarga de los tanques:
      ![](Pasted%20image%2020230612161642.png)

5. Esto probablemente se deba a la asunción de que los volúmenes de control tienen el mismo volumen, y esto depende de dónde realmente estén colocadas las sondas de temperatura, etc. Por lo que el siguiente paso es, partiendo de los UA calibrados, ajustar los volúmenes de control. 👇🥴($T_{ts,c,m}$) Cambiar el volumen afecta a la dinámica de la evolución de la temperatura para cualquier fenómeno incluyendo pérdidas con el ambiente.
   ![](Pasted%20image%2020230613084455.png)
   6. U y V no son parámetros independientes, hay que calibrarlos al mismo tiempo para tener en cuenta el efecto de uno en el otro. Diez mil horas después, se obtiene:
      ![](Pasted%20image%2020230613114347.png)


### Parámetros óptimos

|     | $T_{ts,h,t}$ | $T_{ts,h,m}$ | $T_{ts,h,b}$ | $T_{ts,c,t}$ | $T_{ts,c,m}$ | $T_{ts,c,b}$ |
| --- |:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| UA  |    0.0068    |    0.004     |      -       |    0.0287    |    0.0069    |      -       |
| V   |    2.9722    |    1.7128    |      -       | 9.4346             |     3.7807         |   -           |

### Visualización de energía almacenada en tanques

#### Tmin=84 ºC
![](Pasted%20image%2020230613134058.png)
#### Tmin=80 ºC
![](Pasted%20image%2020230613133944.png)
#### Tmin=60 ºC
![](Pasted%20image%2020230613134036.png)
#### Evolución con el tiempo


## Otros

![Almacenamiento térmico](Validation%20signals%20list.md#Almacenamiento%20térmico)
