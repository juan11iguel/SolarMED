# Changelog

All notable changes to this project will be documented in this file.


## 2024-02 - 2024-03

- Re-written the class implementing the model to make use of pydantic for data validation and serialization
- Improved visualization of model and comparison with experimental data using `phd_visualizations.test_timeseries.experimental_plot` (see [repository](https://github.com/juan11iguel/phd_visualizations))
- New version of model of solar field: no more steady-state approximation, includes variable delay
- New version of inverse solar field model
- New version of heat exchanger model using the effectiveness-NTU method
- Test report with validation visualization for all component models and complete system model in [solarMED_validation_report.ipynb](solarMED_validation_report.ipynb)
- Updated docs
- Re-implementation of the step function of the class to make it more robust and easier to use / understand: now includes system operating state
- Validated with experimental data

Further information can be found in the [documentation](https://github.com/juan11iguel/models_psa/tree/main/docs)

Examples of solar field and solar field inverse model validation:

![solar_field_validation_20231030_beta_1.2416e-02_H_3.408_gamma_0.050](docs/attachments/solar_field_validation_20231030_beta_1.2416e-02_H_3.408_gamma_0.050.svg)
![solar_field_inverse_validation2_20230807_beta_1.1578e-02_H_3.126_gamma_0.047](docs/attachments/solar_field_inverse_validation2_20230807_beta_1.1578e-02_H_3.126_gamma_0.047.svg)


## 2023-09. Solar MED initial version

- First version of the model including solar field and heat exhanger models
- Solar field model
- Coupled subproblem of solar field + heat exchanger + thermal storage
- Initial visualization of the model outputs

## 2023-06. System model initial version

- First a version of the system model with MED and the thermal storage was released...
- MED model
- Thermal storage model
- Three-way valve model