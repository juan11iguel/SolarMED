#TODO: Here is where we should implement all test to validate the correct
working of the model


## Proposed tests

### Test model export import

1. Evaluate a normal model instance (that is, initialized normally) for some 
validation data, evaluate its performance with the `calculate_metrics`
utility function and save the output.
2. An instance of the model should be dumpable using `model.dump_instance()`. Then,
a new instance should be created from the dump with 
`new_model_instace=SolarMED(**model.dump_instance())`, and after being evaluated
the `calculate_metrics` output should be identical to the one produced by the
original instance.