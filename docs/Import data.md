#tutorial

In order to get an operation day data, two sources are currently used:
- Exported txt from legacy SCADA
- Exported csv from librescada db

## Import data from legacy SCADA

In order to generate a csv from the legacy SCADA, there are two options:

- Using a MATLAB function, it should be called like this:

```matlab
datos = procesarDatosSCADA('20230505_tanques_JM.txt', convert_to_double=true);

```

```matlab
writetable(datos, sprintf('%s_solarMED.csv', datestr(datos.TimeStamp(1), 'yyyymmdd')))
```

- Using a feature incomplete (compared to the MATLAB version) Python implementation: `Nextcloud/Juanmi_MED_PSA/Python/labview_txt_parser.ipynb`

## Import data from libreSCADA database

One could access directly the database with the operation data, another option is to use the notebook `Nextcloud/Juanmi_MED_PSA/Python/test_data_to_csv.ipynb`.