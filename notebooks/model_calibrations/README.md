This folder contains different notebooks used to develop and calibrate the individual models of each component that
makes up the SolarMED system.

> ![info] 20241021 Deprecation of inverse variants (solar field and inverse heat generation and storage subproblem)
> Instead of trying to solve inverse models, which greatly increases the complexity of the problem. It has been decided
> to, internally used only direct models. This means that the inputs to the model will be real inputs. As a final step,
> the combined model will be evaluated with those inputs, so any decision variable that is in reality an output of the 
> system can be handled.