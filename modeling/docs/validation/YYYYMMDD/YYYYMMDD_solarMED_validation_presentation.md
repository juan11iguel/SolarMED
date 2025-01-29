---
margin: "0.05"
transition: none
maxScale: "2"
width: "1000"
height: "750"
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
## test YYYYMMDD

---
## Index

- Context
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
### Context. Decision variables

::: block <!-- element style="font-size:8pt-->


|            Variable             | Description                                                       | Unidades | Observations     |
| :-----------------------------: | ----------------------------------------------------------------- | :------: | ---------------- |
|        $T_{med,s,in}^*$         | (float+bool) MED heat source inlet temperature                    |    ºC    |                  |
|          $q_{med,s}^*$          | (float) MED heat source flow rate                                 |   m³/h   |                  |
|          $q_{med,f}^*$          | (float) MED feed water flow rate                                  |   m³/h   |                  |
|        $T_{med,c,out}^*$        | (float) MED condenser outlet temperature                          |    ºC    |                  |
|         $T_{sf,out}^*$          | (float+bool) Solar field outlet temperature                       |    ºC    |                  |
|           $q_{ts}^*$            | (float+bool) Thermal storage heat source re-circulation flow rate |   m³/h   |                  |
| $\textrm{MED}_{vacuum,state}^*$ | (int) MED vacuum system state: 0-OFF, 1-LOW, 2-HIGH               |    -     | Defaults to HIGH |
|                                 | **TOTAL**                                                         |    -     |                  |
:::

--
## Context. Environment

|    Variable    | Descripción          | Unidades |
| :------------: | -------------------- | :------: |
|      $I$       | Solar irradiance     |   W/m²   |
|   $T_{amb}$    | Ambient temperature  |    ºC    |
| $T_{med,c,in}$ | Seawater temperature |    ºC    |

--
## Context. Objective function

- Economic cost function:
$$ J = \min\limits_{\Delta z} \left( C_{e}\left[\frac{u.m.}{kWh}\right]·J_{e} [kW] - C_{w}\left[\frac{u.m.}{m^3} \right]·q_{med,d} \left[ \frac{m^3}{h} \right]\right)·t_{s} \: \left[ u.m. \right] $$

Where the electricity consumption is obtained as the sum of every individual consumption:

$$J_{e} = f(J_{med,vacuum}, J_{med,s}, J_{med,f}, J_{med,c}, J_{med,d}, J_{med,b}, J_{sf}, J_{ts,src})$$

$$ J_{e} = \sum\limits_{i=1}^{N}J_{i} $$

---
<grid drag="100 10" drop="top">
## Test visualization
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_solarMED_visualization.html"></iframe>
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
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_solar_fieldvalidation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
<grid drag="100 10" drop="top">
## Components validation. Solar field results. Flow prediction

More detailed information about the model can be found in the [model documentation](../../models/solar_field.md).
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_solar_field_inverse_validation.html"></iframe>
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
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_heat_exchanger_validation.html"></iframe>
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
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_thermal_storage_validation.html"></iframe>
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
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_MED_validation.html"></iframe>
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
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_solarMED_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--

<grid drag="100 10" drop="top">
## Complete system validation. States evolution.

Result of Finiste State Machines (FSMs) evaluation
</grid>

<grid drop="bottom" drag="100 85">
<iframe width="100%" height="100%" data-src="attachments/YYYYMMDD_solarMED_validation.html"></iframe>
<!-- element style="width:100%; height:100%" -->
</grid>

--
<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step X**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step X**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke"> 
![Operating mode step X](../../operating_modes/state_sf_ACTIVE-ts_ACTIVE-med_ACTIVE.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step X**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step X**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke" frag="X"> 
![Operating mode step X](../../operating_modes/state_sf_IDLE-ts_IDLE-med_OFF.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step X**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step X**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke" frag="X"> 
![Operating mode step X](../../operating_modes/state_sf_ACTIVE-ts_ACTIVE-med_GENERATING_VACUUM.svg) 
</grid>

<grid drag="100 10" drop="top" bg="whitesmoke">
**Operating mode at step X**
</grid>
<grid drag="100 16" drop="0 12" bg="whitesmoke">
**Place holder for plot at step X**
</grid>
<grid drag="100 70" drop="bottom" bg="whitesmoke" frag="X"> 
![Operating mode step X](../../operating_modes/state_sf_ACTIVE-ts_IDLE-med_IDLE.svg) 
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
