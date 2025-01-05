from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import GPy

import warnings

# Hide poorly implemented warning from GPy or a dependency
# It is raised when importing a model according to the instructions:
# RuntimeWarning:Don't forget to initialize by self.initialize_parameter()!
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", 
        category=RuntimeWarning, 
        module=r".*paramz\.parameterized"
    )

@dataclass
class FixedModelParameters:
    """
    MED fixed model parameters.
    Default values could be replaced by values from a configuration file. 
    """
    Tmed_s_min: float = 60  # Minimum operating heat source temperature [ºC]
    Tmed_s_max: float = 75  # Maximum operating heat source temperature [ºC]
    qmed_c_min: float = 8   # Minimum cooling flow rate [m³/h]
    qmed_c_max: float = 21  # Maximum cooling flow rate [m³/h]
    qmed_s_min: float = 30  # Minimum heat source flow rate [m³/h]
    qmed_s_max: float = 48  # Maximum heat source flow rate [m³/h]
    qmed_f_min: float = 5   # Minimum feed flow rate [m³/h]
    qmed_f_max: float = 9   # Maximum feed flow rate [m³/h]

    param_array: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (2.03122253e+00, 5.98578573e+00, 4.47722101e-03),
        (1.81219256e+03, 1.03129028e+02, 1.47705700e-02),
        (133.14646561,   4.93363467,   0.29595623)
        ])

def import_train_data(model_data_path: Path, ydim: int = 3) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """ Import GPR model train data """
    
    gpr_data: np.ndarray[float] = np.load(model_data_path)

    # Split the combined array back into X_train and Y_train (removing the padding)
    X_train = gpr_data[:, :, 0]
    Y_train = gpr_data[:, :, 1][:, :ydim]
    
    return X_train, Y_train
    

class MedModel:
    """
    
    Outputs: qd, Ts,out, qc
    Inputs: Ms, Ts,in, Mf, Tc,out, Tc,in
    
    """
    
    X: np.ndarray[float]
    Y: np.ndarray[float]
    param_array: list[tuple[float, float, float]]
    models: list[GPy.models.GPRegression]
    
    param_array: list[tuple[float, float, float]] = [
        (2.03122253e+00, 5.98578573e+00, 4.47722101e-03),
        (1.81219256e+03, 1.03129028e+02, 1.47705700e-02),
        (133.14646561,   4.93363467,   0.29595623)
    ]
    
    def __init__(self, X: np.ndarray[float] = None, Y: np.ndarray[float] = None, train_data_path: Path = None, 
                 param_array: list[tuple[float, float, float]] = None):
        
        # assert (X is not None and Y is not None) or train_data_path is not None, \
        #     "Either train data (`X` and `Y`) are provided, or a valid (`train_data_path`) path to load them using `import_model_data` needs to be provided"
        
        if train_data_path is not None:
            self.X, self.Y = import_train_data(train_data_path)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X, self.Y = import_train_data(
                Path(__file__).resolve().parent / 'train_data.npy'
            )
        
        if param_array is not None:
            self.param_array = param_array
        self.models = self._create_models()
    
    def _create_models(self) -> list[GPy.models.GPRegression]:
        models = []
        for i, param in enumerate(self.param_array):
            model = GPy.models.GPRegression(self.X, self.Y[:, [i]], initialize=False)
            model.update_model(False) # do not call the underlying expensive algebra on load
            model.initialize_parameter() # Initialize the parameters (connect the parameters up)
            model[:] = param # Load the parameters
            model.update_model(True) # Call the algebra only once
            models.append(model)
        return models
    
    def predict(self, X: np.ndarray, return_variances: bool = False) -> np.ndarray[float] | tuple[np.ndarray[float], np.ndarray[float]]:
        
        Y_preds, Y_pred_vars = zip(*(model.predict(X) for model in self.models))
        Y_preds = np.hstack(Y_preds)  # Combine predictions
        
        if return_variances:
            return Y_preds, np.hstack(Y_pred_vars)  # Combine variances
        return Y_preds