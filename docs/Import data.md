#tutorial

In order to get an operation day data, two sources are currently used:
- Exported txt from legacy SCADA
- Exported csv from librescada db

In order to generate a csv from the legacy SCADA, a MATLAB function is used, it should be called like this:

```matlab
datos = procesarDatosSCADA('20230505_tanques_JM.txt', convert_to_double=true);

```

```matlab
writetable(datos, sprintf('%s_solarMED.csv', datestr(datos.TimeStamp(1), 'yyyymmdd')))
```
