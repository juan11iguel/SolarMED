

---
Generated at 2024-04-04 17:32
Sample rate: 30s

---

# Solar MED model validation report for test 20230703

Validation report for Solar MED model, it includes validation graphs for the individual [component models](#Components) as well as the [complete system](#Complete system).

For the different visualizations, a static version if shown here, but an interactive `html` version is also available that can be opened in any browser. The link to it is shown above the static image.

# Test visualization

[Interactive version](attachments/20230703_solarMED_visualization.html)
![Test representation](attachments/20230703_solarMED_visualization.svg)

# Components

## Solar field

More detailed information about the model can be found in the [model documentation](../../models/solar_field.md).

- Parameters

| Parameter | Value |
| --------- | ----- |
| β (m)     |  4.36e-02     |
| H (W/m²)? |  13.68     |

### Temperature prediction

- Performance metrics

| Metric | Value |
| ------ | ----- |
| IAE    |       |
| RMSE   |       |
| ITAE   |       |
| MAE    |       |


[Interactive version](attachments/20230703_solar_field_validation.html)
![Solar field validation](attachments/20230703_solar_field_validation.svg)

### Inverse (flow prediction)


- Performance metrics

| Metric | Value |
| ------ | ----- |
| IAE    |       |
| RMSE   |       |
| ITAE   |       |
| MAE    |       |

[Interactive version](attachments/20230703_solar_field_inverse_validation.html)
![Solar_field_inverse validation](attachments/20230703_solar_field_inverse_validation.svg)

## Heat exchanger

More detailed information about the model can be found in the [model documentation](../../models/heat_exchanger.md).

- Parameters

| Parameter | Value |
| --------- | ----- |
| UA (W/K)  | 1.35e+04      |
| H (W/m²)? | 0.00     |

- Performance metrics

| Metric | Value |
| ------ | ----- |
| IAE    |       |
| RMSE   |       |
| ITAE   |       |
| MAE    |       |

[Interactive version](attachments/20230703_heat_exchanger_validation.html)
![solar MED validation](attachments/20230703_heat_exchanger_validation.svg)

## Thermal storage

More detailed information about the model can be found in the [model documentation](../../models/thermal_storage.md).

- Parameters

| Parameter    | Top        | Medium     | Bottom     |
| ------------ | ---------- | ---------- | ---------- |
| $UA_h$ (W/K) | 6.98e-03 | 5.84e-03 | 3.04e-02 |
| $V_h$ (m³)   | 5.95 | 4.88 | 2.20 |
| $UA_c$ (W/K) | 1.40e-02 | 1.00e-04 | 2.29e-02 |
| $V_c$ (m³)   | 5.33 | 7.56 | 0.91 |

- Performance metrics

| Metric | Value |
| ------ | ----- |
| IAE    |       |
| RMSE   |       |
| ITAE   |       |
| MAE    |       |

[Interactive version](attachments/20230703_thermal_storage_validation.html)
![Thermal storage validation](attachments/20230703_thermal_storage_validation.svg)

## MED

More detailed information about the model can be found in the [model documentation](../../models/MED.md).

- Parameters


- Performance metrics

| Metric | Value |
| ------ | ----- |
| IAE    |       |
| RMSE   |       |
| ITAE   |       |
| MAE    |       |

[Interactive version](attachments/20230703_MED_validation.html)
![MED validation](attachments/20230703_MED_validation.svg)

# Complete system

More detailed information about the model can be found in the [model documentation](../../models/complete_system.md).

- Parameters

(For some reason the static image generation is broken, but the interactive version is displayed correctly)

[Interactive version](attachments/20230703_solarMED_validation.html)
![solar MED validation](attachments/20230703_solarMED_validation.svg)

