## Restricciones estáticas

#### Entorno

- $T\_{amb,min}=-15:  (^{\\circ}C):$ Temperatura ambiente mínima
- $T\_{amb,max}=50:  (^{\\circ}C):$ Temperatura ambiente máxima (Almería en Julio)
- $HR\_{min}=0 : (%):$ Humedad relativa mínima
- $HR\_{max} = 100 : (%):$ Humedad relativa máxima
- $I\_{min}=0 :(W/m^2) :$ Irradiancia solar mínima
- $I\_{max}=2000? :(W/m^2) :$ Irradiancia solar máxima
- $T\_{med,c,in,min} =10 :  (^{\\circ}C):$ Temperatura mínima del agua de mar / entrada del condensador. Puesto a ojo pero se puede buscar cuál es la temperatura mínima que alcanza el mar mediterráneo por ejemplo.
- $T\_{med,c,in,max} = 28: (^{\\circ}C):$ Temperatura máxima del agua de mar / entrada al condensador. Puesto a ojo pero se puede buscar cuál es la temperatura máxima que alcanza el mar mediterráneo por ejemplo. Igual hay que ir actualizándolo cada año para tener en cuenta cambio climático  🥴

#### MED

- $\\dot{m}\_{c,min} = 10 : (m^3/h):$ Caudal mínimo de agua de refrigeración que puede circular por el condensador
- $\\dot{m}\_{c,max} = 21 : (m^3/h):$ Caudal máximo de agua de refrigeración que puede circular por el condensador
- $\\dot{m}\_{s,min} = 25.2 : (m^3/h), 7 : (l/s):$ Caudal mínimo de agua caliente que puede circular por el primer efecto
- $\\dot{m}\_{s,max} = 50.4 : (m^3/h), 12 : (l/s):$ Caudal máximo de agua caliente que puede circular por el primer efecto
- $\\dot{m}\_{f,min} = 5 : (m^3/h):$ Caudal mínimo de agua de alimentación
- $\\dot{m}\_{f,max} = 9 : (m^3/h):$ Caudal máximo de agua de alimentación

#### Thermal storage

- $\\dot{m}\_{ts,src,min} = - : (m^3/h), - : (l/min):$ Caudal mínimo de agua caliente para calentar almacenamiento térmico / agua que circula por el secundario del intercambiador
- $\\dot{m}\_{ts,src,max} = - : (m^3/h), - : (l/min):$ Caudal máximo de agua caliente para calentar almacenamiento térmico / agua que circula por el secundario del intercambiador

## Restricciones dinámicas

Evolucionan con el tiempo en función de las circunstancias

### Configurable previo a cálculo

Restricción que se puede estimar con las entradas actuales previo a resolver sistema, por lo que una entrada fuera de rango puede ser identificada *a priori*.

### Configurable sólo tras cálculo

Restricción que sólo se puede estimar tras haber resuelto el modelo. Una entrada fuera de rango o que provoca una salida fuera de rango sólo puede ser identificada *a posteriori*.

#### MED

- $T\_{c,out,min} : (^{\\circ}C):$ Temperatura de salida del condensador mínima $$T\_{c,out,min} = T\_{c,in}+\\frac{\\dot{m}*d·\\lambda(T_d)}{\\dot{m}*{c,max}·c\_{p}(T\_{c,in})} $$
- $T\_{c,out,max} : (^{\\circ}C):$ Temperatura de salida del condensador máxima $$T\_{c,out,max} = T\_{c,in}+\\frac{\\dot{m}*d·\\lambda(T_d)}{\\dot{m}*{c,min}·c\_{p}(T\_{c,in})} $$