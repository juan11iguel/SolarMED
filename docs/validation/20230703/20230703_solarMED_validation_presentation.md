---
generated_at: 2024-05-09 17:12
sample_rate: 30s
margin: 0.05
transition: none
maxScale: 2
width: 1000
height: 750
---

---
%% Model parameters: %%
`lims_mts_src`: `(0.95, 20)`
`lims_msf`: `(4.7, 14)`
`lims_mmed_s`: `(30, 48)`
`lims_mmed_f`: `(5, 9)`
`lims_mmed_c`: `(8, 21)`
`lims_Tmed_s_in`: `(60, 75)`
`lims_Tsf_out`: `(65, 120.0)`
`lims_T_hot`: `(0, 120.0)`
`use_models`: `True`
`use_finite_state_machine`: `True`
`sample_time`: `30.0`
`resolution_mode`: `simple`
`vacuum_duration_time`: `300`
`brine_emptying_time`: `180`
`startup_duration_time`: `60`
`med_actuators`: `[{'id': 'med_brine_pump', 'coefficients': [0.010371467694486103, -0.025160600483389525, 0.03393870518526908]}, {'id': 'med_feed_pump', 'coefficients': [0.7035299527191431, -0.09466303549610014, 0.019077706335712326]}, {'id': 'med_distillate_pump', 'coefficients': [4.149635559273511, -3.6572156762250954, 0.9484207971761789]}, {'id': 'med_cooling_pump', 'coefficients': [5.2178993694785625, -0.9238542100009888, 0.056680794931454774]}, {'id': 'med_heatsource_pump', 'coefficients': [0.031175213554380448, -0.01857544733009508, 0.0013320144040346285]}]`
`ts_actuators`: `[{'id': 'ts_src_pump', 'coefficients': [0.0, 0.0, 0.0, 0.0, 0.0]}]`
`UAts_h`: `[0.0069818, 0.00584034, 0.03041486]`
`UAts_c`: `[0.01396848, 0.0001, 0.02286885]`
`Vts_h`: `[5.94771006, 4.87661781, 2.19737023]`
`Vts_c`: `[5.33410037, 7.56470594, 0.90547187]`
`sf_actuators`: `[{'id': 'sf_pump', 'coefficients': [0.0, 0.0, 0.0, 0.0, 0.0]}]`
`beta_sf`: `0.0436396`
`H_sf`: `13.676448551722462`
`gamma_sf`: `0.1`
`filter_sf`: `0.1`
`nt_sf`: `1`
`np_sf`: `35`
`ns_sf`: `2`
`Lt_sf`: `23.0`
`Acs_sf`: `7.85e-05`
`Kp_sf`: `-0.1`
`Ki_sf`: `-0.01`
`UA_hx`: `13536.596`
`H_hx`: `0`

---

<style>
	.with-blue-border{
		border: 1px solid blue;
	}
	
	.with-border{
		border: 1px solid red;
	}

	 .scaled_to_zoom{
		 zoom: 0.7; 
	    -moz-transform: scale(0.7); 
	    -moz-transform-origin: 0 0;
	 }
	 
	 .scaled_iframe{
        -ms-zoom: 0.75;
        -moz-transform: scale(0.75);
        -moz-transform-origin: 0 0;
        -o-transform: scale(0.75);
        -o-transform-origin: 0 0;
        -webkit-transform: scale(0.75);
        -webkit-transform-origin: 0 0;
	}
</style>


# Solar MED model validation report 
## test 20230703

---
## Index

- Context
    - Nomenclature
    - Decision variables description
    - Environment variables description
    - Objective function
- Test visualization
- Components validation
- Solar field
	- Temperature prediction
	- Flow prediction
- Heat exchanger
- Thermal storage
- MED
- Complete system validation
	- Model
	- State evolution visualization

---
## Context.

Poner algún texto introductorio del proceso

--

## Context. Nomenclature

![solarMED_diagram](../../models/attachments/solarMED_optimization-general_diagram.svg)
[Nomenclature](../../Nomenclature.md)

--
### Context. Decision variables description

::: block <!-- element style="font-size:8pt-->


|            Variable             | Description                                                       | Unidades | Observations     |
| :-----------------------------: | ----------------------------------------------------------------- | :------: | ---------------- |
|        $T_{med,s,in}^*$         | (float+bool) MED heat source inlet temperature                    |    ºC    |                  |
|          $q_{med,s}^*$          | (float) MED heat source flow rate                                 |   m³/h   |                  |
|          $q_{med,f}^*$          | (float) MED feed water flow rate                                  |   m³/h   |                  |
|        $T_{med,c,out}^*$        | (float) MED condenser outlet temperature                          |    ºC    |                  |
|         $T_{sf,out}^*$          | (float+bool) Solar field outlet temperature                       |    ºC    |                  |
|           $q_{ts}^*$            | (float+bool) Thermal storage heat source re-circulation flow rate |   m³/h   |                  |
| $	extrm{MED}_{vacuum,state}^*$ | (int) MED vacuum system state: 0-OFF, 1-LOW, 2-HIGH               |    -     | Defaults to HIGH |
|                                 | **TOTAL**                                                         |    -     |                  |
:::

--
### Context. Environment variables description

|    Variable    | Descripción          | Unidades |
| :------------: | -------------------- | :------: |
|      $I$       | Solar irradiance     |   W/m²   |
|   $T_{amb}$    | Ambient temperature  |    ºC    |
| $T_{med,c,in}$ | Seawater temperature |    ºC    |

--
## Context. Objective function

- Economic cost function:
$$ J = \min\limits_{\Delta z} \left( C_{e}\left[rac{u.m.}{kWh}ight]·J_{e} [kW] - C_{w}\left[rac{u.m.}{m^3} ight]·q_{med,d} \left[ rac{m^3}{h} ight]ight)·t_{s} \: \left[ u.m. ight] $$

Where the electricity consumption is obtained as the sum of every individual consumption:

$$J_{e} = f(J_{med,vacuum}, J_{med,s}, J_{med,f}, J_{med,c}, J_{med,d}, J_{med,b}, J_{sf}, J_{ts,src})$$

$$ J_{e} = \sum\limits_{i=1}^{N}J_{i} $$

---
<grid drag="100 10" drop="top">
## Test visualization
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_solarMED_visualization.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

---
Components validation
--
Components validation. Solar field diagram

![Solar field diagram](../../models/attachments/solarMED_optimization-solar_field.drawio.svg)

--
<grid drag="100 10" drop="top">
## Components validation. Solar field results. Temperature prediction

More detailed information about the model can be found in the [model documentation](../../models/solar_field.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_solar_fieldvalidation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
<grid drag="100 10" drop="top">
## Components validation. Solar field results. Flow prediction

More detailed information about the model can be found in the [model documentation](../../models/solar_field.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_solar_field_inverse_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
Components validation. Heat exchanger diagram

![Heat exchanger diagram](../../models/attachments/solarMED_optimization-Heat_exchanger.svg)

--
<grid drag="100 10" drop="top">
## Components validation. Heat exhanger results

More detailed information about the model can be found in the [model documentation](../../models/heat_exchanger.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_heat_exchanger_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
Components validation. Thermal storage diagram

![thermal storage diagram](../../models/attachments/solarMED_optimization-Storage model.drawio.svg)

--
<grid drag="100 10" drop="top">
## Components validation. Thermal storage results

More detailed information about the model can be found in the [model documentation](../../models/thermal_storage.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_thermal_storage_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>
--
Components validation. MED diagram

Pendiente: Poner un diagrama guapete

![thermal storage diagram](../../models/attachments/solarMED_optimization-Storage model.drawio.svg)
--
<grid drag="100 10" drop="top">
## Components validation. MED results

More detailed information about the model can be found in the [model documentation](../../models/MED.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_MED_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

---
# Complete system validation
--
Complete system validation. System diagram

![solarMED_diagram](../../models/attachments/solarMED_optimization-general_diagram.svg)
--
<grid drag="100 10" drop="top">
### Complete system validation. Results

More detailed information about the model can be found in the [model documentation](../../models/complete_system.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_solarMED_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--

<grid drag="100 10" drop="top">
## Complete system validation. States evolution.

Result of Finiste State Machines (FSMs) evaluation
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/20230703_solarMED_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 0 (2023-07-03 07:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 0**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_GENERATING_VACUUM.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 20 (2023-07-03 07:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 20**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 40 (2023-07-03 07:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 40**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 60 (2023-07-03 08:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 60**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 80 (2023-07-03 08:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 80**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 100 (2023-07-03 08:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 100**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 120 (2023-07-03 08:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 120**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 140 (2023-07-03 08:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 140**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 160 (2023-07-03 08:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 160**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 180 (2023-07-03 09:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 180**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_SHUTTING_DOWN.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 200 (2023-07-03 09:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 200**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 220 (2023-07-03 09:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 220**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 240 (2023-07-03 09:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 240**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_STARTING_UP.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 260 (2023-07-03 09:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 260**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 280 (2023-07-03 09:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 280**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 300 (2023-07-03 10:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 300**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 320 (2023-07-03 10:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 320**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 340 (2023-07-03 10:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 340**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 360 (2023-07-03 10:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 360**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 380 (2023-07-03 10:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 380**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 400 (2023-07-03 10:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 400**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 420 (2023-07-03 11:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 420**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 440 (2023-07-03 11:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 440**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 460 (2023-07-03 11:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 460**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 480 (2023-07-03 11:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 480**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 500 (2023-07-03 11:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 500**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 520 (2023-07-03 11:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 520**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 540 (2023-07-03 12:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 540**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 560 (2023-07-03 12:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 560**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 580 (2023-07-03 12:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 580**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 600 (2023-07-03 12:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 600**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 620 (2023-07-03 12:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 620**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 640 (2023-07-03 12:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 640**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 660 (2023-07-03 13:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 660**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 680 (2023-07-03 13:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 680**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 700 (2023-07-03 13:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 700**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 720 (2023-07-03 13:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 720**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 740 (2023-07-03 13:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 740**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 760 (2023-07-03 13:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 760**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 780 (2023-07-03 14:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 780**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 800 (2023-07-03 14:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 800**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 820 (2023-07-03 14:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 820**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 840 (2023-07-03 14:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 840**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 860 (2023-07-03 14:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 860**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 880 (2023-07-03 14:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 880**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 900 (2023-07-03 15:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 900**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 920 (2023-07-03 15:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 920**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 940 (2023-07-03 15:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 940**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 960 (2023-07-03 15:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 960**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 980 (2023-07-03 15:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 980**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1000 (2023-07-03 15:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1000**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1020 (2023-07-03 16:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1020**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1040 (2023-07-03 16:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1040**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_ACTIVE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1060 (2023-07-03 16:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1060**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1080 (2023-07-03 16:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1080**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1100 (2023-07-03 16:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1100**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1120 (2023-07-03 16:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1120**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1140 (2023-07-03 17:00)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1140**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1160 (2023-07-03 17:10)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1160**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1180 (2023-07-03 17:20)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1180**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1200 (2023-07-03 17:30)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1200**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1220 (2023-07-03 17:40)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1220**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step 1240 (2023-07-03 17:50)**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step 1240**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/sf_ACTIVE_ts_IDLE_med_IDLE.svg) 
</grid>


--

### Context. States description
::: block <!-- element class="" style="font-size:14pt;" grid="drop:bottom"-->

### Solar field

- **Idle**. Passive state. In this state no fluid is circulating through the solar field and there are just losses to the environment.
- **Active**. Fluid is circulating through the solar field.

### Thermal storage

- **Idle**. Passive state. In this state is always discharging, it may just be losing heat to the environment (slow discharge), or releasing its heat to the load (fast discharge). 
- **Active**. Recirculating state. In this state water from the tanks gets heated in the heat exchanger.

### MED

- **Off**. No vacuum, nor heat or cooling. No distillate produced.
- **Generating vacuum**. Generating vacuum (high state), no heat nor cooling. No distillate produced.
- **Idle**. Vacuum in low state, plant not using heat nor cooling. No distillate produced
- **Starting-up**. Vacuum in low state, plant using heat and cooling. No distillate produced. Its duration is a function of the initial starting point.
- **Active**. Vacuum in low state and plant using heat and cooling to produce distillate.
- **Shutting-down**. Vacuum off, plant not using heat or cooling, no distillate produced, brine pump active.
:::

