
## Campo solar

|    Lazo     | Nombre          | ID         |  Unidades   |
| :---------: | --------------- | ---------- | :---------: |
|    **2**    | $T_{sf,l2,in}$  | TT-SF-018  | $^{\circ}C$ |
|             | $T_{sf,l2,out}$ | TT-SF-019  | $^{\circ}C$ |
|             | $q_{sf,l2}$     | FT-SF-005  |             |
|    **3**    | $T_{sf,l3,in}$  | TT-SF-024  | $^{\circ}C$ |
|             | $T_{sf,l3,out}$ | TT-SF-025  | $^{\circ}C$ |
|             | $q_{sf,l3}$     | FT-SF-006  |             |
|    **4**    | $T_{sf,l4,in}$  | TT-SF-030  | $^{\circ}C$ |
|             | $T_{sf,l4,out}$ | TT-SF-031  | $^{\circ}C$ |
|             | $q_{sf,l4}$     | FT-SF-007  |             |
|    **5**    | $T_{sf,l5,in}$  | TT-SF-036  | $^{\circ}C$ |
|             | $T_{sf,l5,out}$ | TT-SF-037  | $^{\circ}C$ |
|             | $q_{sf,l5}$     | FT-SF-008  |             |
| **Global**  |                 |            |             |
|             | $q_{sf}$        | FT-SF-002  |             |
|             | $T_{sf,in}$     | TT-SF-009  | $^{\circ}C$ |
|             | $T_{sf,out}$    | TT-SF-010  | $^{\circ}C$ |
| **Entorno** |                 |            |             |
|             | $T_{amb}$       | TT-DES-030 | $^{\circ}C$ |
|             | I?              | RE-SF-001  |      ?      |

![](Pasted%20image%2020230622133339.png)

A partir de fichero txt de SCADA:

```MATLAB
datos = procesarDatosSCADA('datos/datos_campo.txt', "listado_variables", ["TT-SF-018", "TT-SF-019", "TT-SF-024", "TT-SF-025", "TT-SF-030", "TT-SF-031", "TT-SF-036", "TT-SF-037", "FT-SF-005", "FT-SF-006", "FT-SF-007", "FT-SF-008", "FT-SF-002", "TT-SF-009", "TT-SF-010", "TT-DES-030", "RE-SF-001"], "filtrarNoOp", "FT-SF-002", "filtrarNoOp_threshold", 1);

%datos=datos(4:end, :);

writetable(datos, 'datos/datos_campo.csv')
```

## Intercambiador campo

| Nombre         | ID         |  Unidades   |
| -------------- | ---------- | :---------: |
| $T_{hx,p,in}$  | TT-SF-010  | $^{\circ}C$ |
| $T_{hx,p,out}$ | TT-SF-009  | $^{\circ}C$ |
| $T_{hx,s,in}$  | TT-SF-007  | $^{\circ}C$ |
| $T_{hx,s,out}$ | TT-SF-008  | $^{\circ}C$ |
| $T_{amb}$      | TT-DES-030 | $^{\circ}C$ |
| $q_{hx,p}$     | FT-SF-002  |             |
| $q_{hx,s}$     | FT-SF-001  |   $L/min$   |

![](Pasted%20image%2020230622131131.png)

A partir de fichero txt de SCADA:

```MATLAB
datos = procesarDatosSCADA('datos/datos_intercambiador.txt', "listado_variables", ["TT-SF-010", "TT-SF-009", "TT-SF-007", "TT-SF-008", "TT-DES-030", "FT-SF-002", "FT-SF-001"], "filtrarNoOp", "FT-SF-002", "filtrarNoOp_threshold", 1);

%datos=datos(4:end, :);

writetable(datos, 'datos/datos_intercambiador.csv')
```

## Almacenamiento térmico

| Nombre             | ID             |  Unidades   |
| ------------------ | -------------- | :---------: |
| $T_{ts,h,out}$     | **TT-AQU-106** | $^{\circ}C$ |
| $T_{ts,h,t}$       | TT-SF-004      | $^{\circ}C$ |
| $T_{ts,h,m}$       | TT-SF-005      | $^{\circ}C$ |
| $T_{ts,h,b}$       | TT-SF-006      | $^{\circ}C$ |
| $T_{ts,c,t}$       | TT-SF-001      | $^{\circ}C$ |
| $T_{ts,c,m}$       | TT-SF-002      | $^{\circ}C$ |
| $T_{ts,c,b}$       | TT-SF-003      | $^{\circ}C$ |
| $T_{amb}$          | TT-DES-030     | $^{\circ}C$ |
| $\dot{m}_{ts,src}$ | FT-SF-001      |    L/min    |
| $\dot{m}_{ts,dis}$ | FT-AQU-100     |     L/s     |
| $T_{ts,t,in}$      | TT-SF-008      | $^{\circ}C$ |
| $T_{ts,c,b,in}$    | TT-AQU-109     | $^{\circ}C$ |
| $T_{ts,c,t,in}$    | **TT-SF-006**  | $^{\circ}C$ |
| $T_{ts,h,b,in}$    | **TT-SF-001**  | $^{\circ}C$ |
| $T_{ts,c,out}$     | **TT-SF-007**  | $^{\circ}C$ |
| $R_{3wv}$          | ZT-AQU-TCV102  |    $\%$     |

A partir de fichero txt de SCADA:

```MATLAB

datos = procesarDatosSCADA('datos/datos_tanques.txt', "listado_variables", ["TT-SF-004", "TT-SF-005", "TT-SF-006", "TT-SF-001", "TT-SF-002", "TT-SF-003", "TT-DES-030", "FT-SF-001", "FT-AQU-100", "TT-SF-008", "TT-AQU-109", "FT-AQU-101", "TT-AQU-106", "TT-SF-007", "ZT-AQU-TCV102"], "filtrarNoOp", "TT-SF-004", "filtrarNoOp_threshold", 20);

datos=datos(4:end, :);

writetable(datos, 'datos/datos_tanques.csv')
```

## Válvula de tres vías

|       Nombre        |           ID            |  Unidades   |
| :-----------------: | :---------------------: | :---------: |
|    $T_{3wv,src}$    |        TT-SF-004        | $^{\circ}C$ |
|  $T_{3wv,dis,in}$   | ~~HW1TT20~~ TT-AQU-107a | $^{\circ}C$ |
|  $T_{3wv,dis,out}$  |         HW1TT21         | $^{\circ}C$ |
|      $R_{3wv}$      |      ZT-AQU-TCV102      |    $\%$     |
| $\dot{m}_{3wv,src}$ |       FT-AQU-101        |     L/s     |
| $\dot{m}_{3wv,dis}$ |       FT-AQU-100        |     L/s     |

![500](Pasted%20image%2020230622130846.png)

A partir de fichero txt de SCADA:

```MATLAB
datos = procesarDatosSCADA('datos/20230621_valvula.txt', "listado_variables", ["TT-SF-004", "TT-AQU-107a", "HW1TT21", "FT-AQU-101", "FT-AQU-100", "ZC-AQU-TCV102", "TT-AQU-109", "TT-AQU-106"], "filtrarNoOp", "FT-AQU-100", "filtrarNoOp_threshold", 1);

writetable(datos, 'datos/datos_valvula.csv')
```

### Auxiliar

Ajustes de curvas de caudal para distintas bombas con relación a la frecuencia

1. Leer datos

```
datos = procesarDatosSCADA('datos/20230707_20230710_tanquesJM.txt', "filtrarNoOp", "FT-AQU-101", "filtrarNoOp_threshold", 0.0001, "listado_variables", ["FT-AQU-100", "FT-AQU-101", "ZT-AQU-TCV102", "SC-AQU-P102", "UK-SF-P001-fq", "FT-SF-001"]);
```

#### $\dot{m}_s$

1. Representar cuando la bomba se esté usando

```
filter = datos.("SC-AQU-P102") > 1; plot(datos.TimeStamp(filter), datos.("SC-AQU-P102")(filter)); hold on; plot(datos.TimeStamp(filter), datos.("FT-AQU-100")(filter)); plot(datos.TimeStamp(filter), datos.("ZT-AQU-TCV102")(filter))
```

1. Manualmente seleccionar tramo donde se han hecho trenes de escalones (Tramo 20230707 08:48:40 - 20230707 09:16:29)
2. Ajustar curva

#### $\dot{m}_{ts,dis}$