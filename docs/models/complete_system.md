# Complete system model

The complete system is obtained through the combination of all the different component models by connecting them properly, as in the system diagram:

![solarMED_optimization-general_diagram](attachments/solarMED_optimization-general_diagram.svg)


Some technical details:

- The complete system is implemented as a Python class in the module `models_psa/solar_med.py`
- It makes use of Pydantic, which facilitates the efficient handling of all validations, parsing and so on. 

## Inputs / outputs flow diagram

> [!warning] Deprecated diagram
> Outdated nomenclature, some outputs have changed, but it's still gives an idea of the information flow so it's kept here

![center | (DEPRECATED) solarMED_optimization-Complete system model.drawio](attachments/solarMED_optimization-Complete%20system%20model.drawio.svg)

## Nomenclature

![solarMED_optimization-general_diagram](attachments/solarMED_optimization-general_diagram.svg)

The nomenclature is maintained with respect to each component, but the component key is prefixed to each variable, e.g. in the case of the MED: $T_{s,in} \rightarrow T_{med,s,in}$. All the variables are defined in the class initialization and contain units and description, checking them there guarantees to be visualizing the latest version. 

%%
- MED 
  - $T_{med,s,in}$
  - $T_{med,cw,out}$
  - $\dot{m}_{med,s}$
  - $\dot{m}_{med,f}$
- Solar field 
  - $T_{sf,out}$
  - $\dot{m}_{sf}$
- Thermal storage 
  - $T_{ts,t,in}$
  - $T_{ts,t}$
  - $T_{ts,b}$
  - 
%%
### Inputs / outputs

Since the model is implemented as a class, once it is initialized, a new iteration can be evaluated by calling the `step` method, it won't return any outputs, but it will update all the internal states and values, which can be accessed individually, or they can be exported as a dataframe with the `to_dataframe` method.

#### Initialization

```python
from solarMED_modeling.solar_med import SolarMED

idx_start = 20
span = 12
idx_end = len(df)
df_mod = pd.DataFrame()

# Initialize model  
model = SolarMED(
    sample_time=60,  # seconds  
    resolution_mode='simple',

    # Initial states  
    ## Thermal storage    
    Tts_h=[df['Tts_h_t'].iloc[idx_start], df['Tts_h_m'].iloc[idx_start], df['Tts_h_b'].iloc[idx_start]],
    Tts_c=[df['Tts_c_t'].iloc[idx_start], df['Tts_c_m'].iloc[idx_start], df['Tts_c_b'].iloc[idx_start]],

    ## Solar field  
    Tsf_in_ant=df['Tsf_in'].iloc[idx_start - span:idx_start].values,
    msf_ant=df['qsf'].iloc[idx_start - span:idx_start].values,

    cost_w=3,  # €/m³   
    cost_e=0.05  # €/kWhe  
)
```

#### Evaluate an/multiple iteration/s

$$ f(q_{med,s}^*, q_{med,f}^*, T_{med,s,in}^*, T_{med,c,out}^*, q_{ts,src}^*, T_{sf,out}^*, T_{med,c,in}, T_{amb}, I) $$

```python
for idx in range(idx_start, idx_end):  

    ds = df.iloc[idx]  
          
    model.step(  
        # Decision variables  
        ## MED        
        mmed_s=ds['qmed_s'],  
        mmed_f=ds['qmed_f'],  
        Tmed_s_in=ds['Tmed_s_in'],  
        Tmed_c_out=ds['Tmed_c_out'],  
        ## Thermal storage  
        mts_src=ds['qhx_s'],  
        ## Solar field  
        Tsf_out=ds['Tsf_out'],  
          
        # Inputs  
        # When the solar field is starting up, a flow can be provided to sync the model with the real system, if a valid Tsf_out is provided, it will be prioritized        
        msf=ds['qsf'] if ds['qsf'] > 4 else None,  
          
        # Environment variables  
        Tmed_c_in=ds['Tmed_c_in'],  
        Tamb=ds['Tamb'],  
        I=ds['I'],  
    )
```
#### Export iteration results

After each `step` call, calling `.to_daframe()` will add a new row in the results dataframe :

```python
df_mod = model.to_dataframe(df_mod, rename_flows=True)
```

#### Evaluate cost function

> [!warning] Not yet implemented at time of writing this page 20240405

After each iteration, the cost function can be evaluated and it will return its current value:

```python
cost = model.evaluate_cost()
```

# Results

In the [validation](../validation) there are available multiple validation reports of the complete model using experimental data.

[Interactive version](../attachments/SolarMED_validation_20231030.html)
![SolarMED_validation_20231030](../attachments/SolarMED_validation_20231030.png)