![](attachments/Pasted%20image%2020240130104058.png)
Haciendo zoom a la parte del modelo:

![](attachments/Pasted%20image%2020240130104144.png)

Parece que el principal contribuidor es `least_squares.py`, en segundo lugar es la llamada a la red neuronal entrenada en MATLAB.

Pero no creo que el problema esté en sí en `least_squares`, si no en el hecho de que para cada iteración está teniendo que evaluar:

- `thermal_storage_model_two_tanks`
- `heat_exchanger_model`
- `solar_field_model`

Lo suyo sería estudiar cada función para ver cuál hay que optimizar, antes de hacer pruebas se me ocurren dos cosas:

- En todas las funciones se están calculando las propiedades del agua (calor específico y densidad), si no hay cambio de estado y los cambios de temperatura son pequeños, igual se podrían simplificar asumiendo propiedades constantes
- Internamente el modelo del tanque hace llamadas a `fsolve`: `initial_guess = Ti_ant`. `Ti = scipy.optimize.fsolve(model_function, initial_guess)`, probablemente esto sea un cuello de botella.

Estas pruebas se pueden incluir en los scripts de model calibration, comparando modelo con propiedades variables y sin serlo, tiempo de ejecución, ajuste, etc.