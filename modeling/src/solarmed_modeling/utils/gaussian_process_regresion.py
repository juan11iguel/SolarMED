from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import GPy

from phd_visualizations.regression import regression_plot

def import_train_data(model_data_path: Path, ydim: int) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """ Import GPR model train data """
    
    gpr_data: np.ndarray[float] = np.load(model_data_path)

    # Split the combined array back into X_train and Y_train (removing the padding)
    X_train = gpr_data[:, :, 0]
    Y_train = gpr_data[:, :, 1][:, :ydim]
    
    return X_train, Y_train
    
@dataclass
class GenericModelParameters:
    """
    Fixed model parameters.
    """
    param_array: list[tuple[float, float, float]]

@dataclass
class GenericModel:
    """
    MED GPR model wrapper.
    Outputs: qd, Ts,out, qc
    Inputs: Ms, Ts,in, Mf, Tc,in, Tc,out
    """
    
    n_outputs: int
    X: Optional[np.ndarray[float]] = None
    Y: Optional[np.ndarray[float]] = None
    models: Optional[list[GPy.models.GPRegression]] = None
    train_data_path: Optional[Path] = None
    fmp: GenericModelParameters = field(default_factory=lambda: GenericModelParameters())
    
    
    def __post_init__(self):
        
        if self.X is not None and self.Y is not None:
            if self.X.shape[0] != self.Y.shape[0]:
                raise ValueError("X and Y must have the same number of samples.")

        elif self.train_data_path is not None:
            self.X, self.Y = import_train_data(self.train_data_path, ydim=self.n_outputs)
        else:
            self.X, self.Y = import_train_data(
                Path(__file__).resolve().parent / 'gpr_data.npy'
            )
        
        self.models = self._create_models()
    
    def _create_models(self) -> list[GPy.models.GPRegression]:
        """
        Create GPR models for each output using the provided training data and fixed model parameters.
        """
        models = []
        for i, param in enumerate(self.fmp.param_array):
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


def train_model(
    data: pd.DataFrame, 
    input_ids: list[str], 
    output_ids: list[str], 
    model_input_ids: list[str], 
    model_output_path: Path,
    train_indices: Optional[np.ndarray[int]] = None,
    val_indices: Optional[np.ndarray[int]] = None,
) -> tuple[list[tuple[float, float, float]], GenericModel, dict[str, pd.DataFrame]]:
    """ Train a GPR model and save the training data """
    
    n_outputs = len(output_ids)

    for var_id in input_ids + output_ids + model_input_ids:
        print(f"{var_id}: {data[var_id].min()} - {data[var_id].max()}")
        
    # Divide data between calibration / training and validation
    X = data[model_input_ids].values
    Y = data[output_ids].values

    # Shuffle the indices
    if train_indices is None or val_indices is None:
        indices = np.arange(len(X))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)

        # Define split point
        split_index = int(0.8 * len(X))  # 80% for training, 20% for validation

        # Split data
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]

    print(f"{X.shape=}, {Y.shape=}")
    print(f"{X_train.shape=}, {Y_train.shape=}")
    print(f"{X_val.shape=}, {Y_val.shape=}")

    # output: (n pts, len(model_input_ids), 2)
    np.save(model_output_path, np.stack(
        (X_train, np.pad(Y_train, ( (0, 0), (0, len(model_input_ids)-len(output_ids)) ))), 
        axis=-1
    ))
    logger.info(f"Saved training data to {model_output_path}")

    # Calibrate a model
    kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1.0, lengthscale=1.0)  # Radial Basis Function kernel

    Y_preds = []
    Y_pred_vars = []
    models = []
    params_arrays = []
    for i in range(Y_train.shape[1]):
        model = GPy.models.GPRegression(X_train, Y_train[:, [i]], kernel.copy())
        model.optimize(messages=True)  # This will optimize the hyperparameters
        models.append(model)
        Y_pred, Y_pred_var = model.predict(X_val)
        Y_preds.append(Y_pred)
        Y_pred_vars.append(Y_pred_var)
        params_arrays.append(model.param_array.tolist())
        
        # Print model summary and params
        print(f"Model for output '{output_ids[i]}':")
        print(model)
        print(f"{model.param_array=}")

    params_arrays = [tuple(arr) for arr in params_arrays]
    print(f"{params_arrays=}")

    Y_preds = np.hstack(Y_preds)  # Combine predictions
    Y_pred_vars = np.hstack(Y_pred_vars)  # Combine variances

    # model = GPy.models.GPRegression(X_train, Y_train, kernel)
    # model.optimize(messages=True)  # This will optimize the hyperparameters

    # Visualize fit

    # Create traces for the plot
    fig = make_subplots(rows=len(output_ids), cols=1, shared_xaxes=True, 
                        subplot_titles=output_ids, x_title="Data Index",
                        vertical_spacing=0.05)

    for idx, output_id in enumerate(output_ids):
        # Add true values as a scatter plot
        fig.add_trace(go.Scatter(
            x=np.arange(Y_val.shape[0]),
            y=Y_val[:, idx],
            mode='markers',
            name='True values',
            showlegend=True if idx==0 else False,
            marker=dict(color='blue')
        ), row=idx+1, col=1)

        # Add predicted values as a line plot
        fig.add_trace(go.Scatter(
            x=np.arange(Y_preds[:, idx].shape[0]),
            y=Y_preds[:, idx],
            mode='lines',
            name='Predicted values',
            showlegend=True if idx==0 else False,
            line=dict(color='red')
        ), row=idx+1, col=1)

        # Add confidence interval as a shaded area
        fig.add_trace(go.Scatter(
            x=np.concatenate([np.arange(Y_preds[:,idx].shape[0]), np.arange(Y_preds[:,idx].shape[0])[::-1]]),
            y=np.concatenate([Y_preds[:, idx] - 2 * np.sqrt(Y_pred_vars[:, idx]), 
                            (Y_preds[:, idx] + 2 * np.sqrt(Y_pred_vars[:, idx]))[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Confidence interval (95%)',
            showlegend=True if idx==0 else False,
        ), row=idx+1, col=1)

    # Update layout
    fig.update_layout(
        title="Regression Results",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        template="plotly_white",
        height=600,
        margin=dict(t=100)
    )

    # Show the plot
    fig.show()

    # Visualization 2. 
    # Create a figure
    fig = go.Figure()

    # Iterate over each output variable
    for i, output_name in enumerate(output_ids):
        # Calculate correlation coefficient
        correlation = np.corrcoef(Y_val[:, i], Y_preds[:, i])[0, 1]
        
        # Add scatter plot for true vs. predicted
        fig.add_trace(go.Scatter(
            x=Y_val[:, i],  # True outputs for the i-th variable
            y=Y_preds[:, i],  # Predicted outputs for the i-th variable
            mode='markers',
            name=f'{output_name} (R²={correlation**2:.2f})',
            marker=dict(size=6, opacity=0.7),
            showlegend=True
        ))

        # Add perfect correlation line (y=x)
        max_val = max(Y_val[:, i].max(), Y_preds[:, i].max())
        min_val = min(Y_val[:, i].min(), Y_preds[:, i].min())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name=f'{output_name} Perfect Correlation (y=x)',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title="True vs Predicted Outputs with Correlation Coefficients",
        xaxis_title="True Outputs",
        yaxis_title="Predicted Outputs",
        legend=dict(yanchor="top", y=0.3, xanchor="left", x=0.7),
        template="plotly_white"
    )

    # Show the plot
    fig.show()

    fig = regression_plot(
        df_ref=pd.DataFrame(Y_val, columns=output_ids),
        df_mod=pd.DataFrame(Y_preds, columns=output_ids),
        show_error_metrics=["r2", "mae"],
        var_ids=output_ids,
        legend_pos="side",
        title_y=0.99,
        figure_layout="horizontal",
        width=1200, height=400
    )
    
    fig.show()
    
    # Evaluate model using object
    
    # Both options should work
    fmp = GenericModelParameters(param_array=params_arrays)
    X,Y = import_train_data(model_output_path, ydim=len(output_ids))
    model = GenericModel(X=X, Y=Y, fmp=fmp, n_outputs=n_outputs) # Provide X and Y
    model = GenericModel(train_data_path=model_output_path, fmp=fmp, n_outputs=n_outputs) # Provide path to load X and Y

    # Generate datasets
    dfs_dict = {
        "ref_val": pd.DataFrame(Y_val, columns=output_ids, index=val_indices),
        "ref_train": pd.DataFrame(Y_train, columns=output_ids, index=train_indices),
        "out_val": pd.DataFrame(
            model.predict(X_val, return_variances=False), 
            columns=output_ids, index=val_indices
        ),
        "out_train": pd.DataFrame(
            model.predict(X_train, return_variances=False), 
            columns=output_ids, index=train_indices
        ),
    }
    
    return params_arrays, model, dfs_dict
