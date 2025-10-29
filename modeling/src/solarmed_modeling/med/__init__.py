from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import GPy

import warnings
# Hide poorly implemented warning from GPy or a dependency
# It is raised when importing a model according to the instructions:
# RuntimeWarning:Don't forget to initialize by self.initialize_parameter()!
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    module=r".*paramz\.parameterized"
)

supported_eval_alternatives = ["standard"]

def import_train_data(model_data_path: Path, ydim: int = 3) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """ Import GPR model train data """
    
    gpr_data: np.ndarray[float] = np.load(model_data_path)

    # Split the combined array back into X_train and Y_train (removing the padding)
    X_train = gpr_data[:, :, 0]
    Y_train = gpr_data[:, :, 1][:, :ydim]
    
    return X_train, Y_train
    
@dataclass
class FixedModelParameters:
    """
    MED fixed model parameters.
    Default values could be replaced by values from a configuration file. 
    """
    
    Tmed_s_min: float = 55  # Minimum operating heat source temperature [ºC]
    Tmed_s_max: float = 81  # Maximum operating heat source temperature [ºC]
    qmed_c_min: float = 3   # Minimum cooling flow rate [m³/h]
    qmed_c_max: float = 22  # Maximum cooling flow rate [m³/h]
    qmed_s_min: float = 30  # Minimum heat source flow rate [m³/h]
    qmed_s_max: float = 48  # Maximum heat source flow rate [m³/h]
    qmed_f_min: float = 5   # Minimum feed flow rate [m³/h]
    qmed_f_max: float = 9   # Maximum feed flow rate [m³/h]
    deltaTc_range: tuple[float, float] = (1.5, 25)  # Condenser inlet-outlet temperature difference range [°C]

    param_array: list[tuple[float, float, float]] = field(default_factory=lambda: [(1.9867205543419775, 6.693755214726311, 0.004159348871379535), (999.8787080345198, 27.55730316759485, 0.011559108440227887), (116.61624543437144, 5.938738928616871, 0.39178961211905033)])
    param_array_aux: list[tuple[float, float, float]] = field(default_factory=lambda: [(385.2097576747307, 21.78824050066122, 0.14051420169703088)])

@dataclass
class MedModel:
    """
    MED GPR model wrapper.
    Outputs: qd, Ts,out, qc
    Inputs: Ms, Ts,in, Mf, Tc,in, Tc,out
    """
    
    models: Optional[list[GPy.models.GPRegression]] = None
    models_aux: Optional[list[GPy.models.GPRegression]] = None
    train_data_path: Optional[Path] = None
    fmp: FixedModelParameters = field(default_factory=lambda: FixedModelParameters())
    n_outputs: int = 3
    
    def __post_init__(self):
        
        if self.train_data_path is None:
            self.train_data_path = Path(__file__).resolve().parent 
            
        X, Y = import_train_data(self.train_data_path / 'med_gpr_data.npy', ydim=self.n_outputs)
        Xaux, Yaux = import_train_data(self.train_data_path / 'med_gpr_data_aux.npy', ydim=1)            
        
        self.models = self._create_models(self.fmp.param_array, X, Y)
        self.models_aux = self._create_models(self.fmp.param_array_aux, Xaux, Yaux)
    
    def _create_models(self, param_array: list[tuple[float, ...]], X: np.ndarray[float], Y: np.ndarray[float]) -> list[GPy.models.GPRegression]:
        """
        Create GPR models for each output using the provided training data and fixed model parameters.
        """
        models = []
        for i, param in enumerate(param_array):
            model = GPy.models.GPRegression(X, Y[:, [i]], initialize=False)
            model.update_model(False) # do not call the underlying expensive algebra on load
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.initialize_parameter() # Initialize the parameters (connect the parameters up)
            model[:] = param # Load the parameters
            model.update_model(True) # Call the algebra only once
            models.append(model)
            
        return models
    
    def predict(self, X: np.ndarray, return_variances: bool = False, aux: bool = False) -> np.ndarray[float] | tuple[np.ndarray[float], np.ndarray[float]]:
        
        models = self.models if not aux else self.models_aux
        Y_preds, Y_pred_vars = zip(*(model.predict(X) for model in models))
        Y_preds = np.hstack(Y_preds)  # Combine predictions
        
        if return_variances:
            return Y_preds, np.hstack(Y_pred_vars)  # Combine variances
        return Y_preds
    
    def __call__(self, qs_m3h: float, Ts_in_C: float, qf_m3h: float, Tc_in_C: float, Tc_out_C: float, debug: bool = False) -> tuple[float, float, float, float]:
        """
        Evaluate the model for given inputs.
        
        Returns:
            Tuple containing (qd_m3h, Ts_out_C, qc_m3h, Pvc_mbar)
        """
                
        if (
            qf_m3h > 0 and 
            qs_m3h > 0 and 
            Ts_in_C > 0 # and
            # (Tc_in_C+self.fmp.deltaTc_range[0] < Tc_out_C < Tc_in_C+self.fmp.deltaTc_range[1]) 
        ):
            
            qd_m3h, Ts_out_C, qc_m3h = self.predict(
                np.array([[qs_m3h, Ts_in_C, qf_m3h, Tc_in_C, Tc_out_C]])    
            ).flatten()
            
            qc_m3h -= 5.

            # qc_m3h = qd_m3h * (4.18 * (Tc_in_C - Tc_out_C)) / (w_props(P=Pvc_mbar/1000, x=0).h -w_props(P=Pvc_mbar/1000, x=1).h)
            # if debug:
            #     return qd_m3h, Ts_out_C, qc_m3h, Pvc_mbar, qc_m3h_gpr
            
            if self.fmp.qmed_c_min <= qc_m3h <= self.fmp.qmed_c_max:
                return qd_m3h, Ts_out_C, qc_m3h, Tc_out_C # Model solved
            
            if qc_m3h > self.fmp.qmed_c_max:
                Tc_out_C = self.predict(
                    np.array([[qs_m3h, Ts_in_C, qf_m3h, Tc_in_C, self.fmp.qmed_c_max]]
                ), aux=True).flatten()[0]
            elif qc_m3h < self.fmp.qmed_c_min:
                Tc_out_C = self.predict(
                    np.array([[qs_m3h, Ts_in_C, qf_m3h, Tc_in_C, self.fmp.qmed_c_min]]
                ), aux=True).flatten()[0]
            
            qd_m3h, Ts_out_C, qc_m3h = self.predict(
                np.array([[qs_m3h, Ts_in_C, qf_m3h, Tc_in_C, Tc_out_C]])    
            ).flatten()
            
            return qd_m3h, Ts_out_C, qc_m3h, Tc_out_C
            
        else:
            return [np.nan]*(self.n_outputs+1)