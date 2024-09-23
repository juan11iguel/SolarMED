# /// script
# dependencies = [
#   "tensorflow>=2.17.0",
#   "scikit-learn>=0.24.2",
#   "numpy>=1.19.5",
#   "pandas>=1.1.5",
# ]
# ///

"""
Class to import models, give predictions, and provide performances.
"""

import pickle
import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd


class Predictor:
    """Predictor imports the models.
    Methods exist to predict on new datasets
    and to get performances on the predictions.
    """

    def __init__(self,
                 full_dataset,
                 path_to_ann_model='./models/',
                 filename_ann_model='ann_model.keras',
                 path_to_gbr_model='./models/',
                 prefix_gbr_model='gbr_model_',
                 filename_gbr_model='.pickle'):
        """Initializes instance, imports models.

        Make sure the GBR models have Mc, Md and Ts in their name.
        Dataset is necessary because ANN models were trained on normalized
        targets, due to differences in magnitude.

        Parameters
        ----------
        path_to_ann_model : str, optional
            relative path to ann model file, by default './'
        filename_ann_model : str, optional
            filename of the ann model file, by default 'optimal_model.keras'
        path_to_gbr_model : str, optional
            relative path to gbr model files, by default './'
        prefix_gbr_model : str, optional
            prefix in gbr model names, by default 'gbr_model_'
        filename_gbr_model : str, optional
            filename of the gbr model files, by default '_model.pickle'
            make sure the filename is preceded by Mc, Md or Ts
        """

        self.target_dataset = full_dataset.loc[:, ['Md', 'Ts_out', 'Mc']]
        self.target_means = self.target_dataset.mean(axis=0)
        self.target_stds = self.target_dataset.std(axis=0)

        self.ann_model = tf.keras.models.load_model(
            f'{path_to_ann_model}{filename_ann_model}')
        with open(f'{path_to_gbr_model}{prefix_gbr_model}Mc{filename_gbr_model}',
                  'rb') as picklefile:
            self.gbr_model_mc = pickle.load(picklefile)
        with open(f'{path_to_gbr_model}{prefix_gbr_model}Md{filename_gbr_model}',
                  'rb') as picklefile:
            self.gbr_model_md = pickle.load(picklefile)
        with open(f'{path_to_gbr_model}{prefix_gbr_model}Ts{filename_gbr_model}',
                  'rb') as picklefile:
            self.gbr_model_ts = pickle.load(picklefile)

    def predict_ann(self, prediction_data):
        """Predicts on data using the ANN model.

        Parameters
        ----------
        prediction_data : pd.DataFrame or np.array
            Two-dimensional dataframe containing data to be predicted with

        Returns
        -------
        np.array
            Two-dimensional array containing predictions
        """

        assert prediction_data.ndim == 2, "Use two-dimensional data as input"
        feature_columns = ['Ms', 'Ts_in', 'Mf', 'Tc_in', 'Tc_out']
        prediction_dataset = prediction_data.loc[:, feature_columns]
        prediction_dataset = prediction_dataset.to_numpy().astype(float)
        prediction = self.ann_model.predict(prediction_dataset)
        prediction = (prediction * np.array(self.target_stds).reshape(1, -1)) + \
            np.array(self.target_means)
        return prediction

    def predict_gbr(self, prediction_data,
                    predict_md=True,
                    predict_ts=True,
                    predict_mc=True):
        """Predicts on data using the ANN model.

        Parameters
        ----------
        prediction_data : pd.DataFrame or np.array
            Two-dimensional dataframe containing data to be predicted with
        predict_X : bool, default=True
            turn off or on whether to predict parameter X.

        Returns
        -------
        np.array
            Two-dimensional array containing predictions
        """

        assert prediction_data.ndim == 2, "Use two-dimensional data as input"
        feature_columns = ['Ms', 'Ts_in', 'Mf', 'Tc_in', 'Tc_out']
        prediction_dataset = prediction_data.loc[:, feature_columns]
        combined_prediction = np.empty((prediction_dataset.shape[0], 0))
        if predict_md:
            md_prediction = self.gbr_model_md.predict(prediction_dataset)
            md_prediction = md_prediction.reshape(-1, 1)
            combined_prediction = np.append(combined_prediction, md_prediction, axis=1)
        if predict_ts:
            ts_prediction = self.gbr_model_ts.predict(prediction_dataset)
            ts_prediction = ts_prediction.reshape(-1, 1)
            combined_prediction = np.append(combined_prediction, ts_prediction, axis=1)
        if predict_mc:
            mc_prediction = self.gbr_model_mc.predict(prediction_dataset)
            mc_prediction = mc_prediction.reshape(-1, 1)
            combined_prediction = np.append(combined_prediction, mc_prediction, axis=1)
        return combined_prediction

    def get_performance_r2(self, predictions, targets):
        """Calculate r2 value on predictions.

        Parameters
        ----------
        predictions : pd.DataFrame or np.array
            Two-dimensional data structure containing predictions
        targets : pd.DataFrame or np.array
            Two-dimensional data structure containing targets

        Returns
        -------
        float
            Performance r2 value
        """
        r2_value = sklearn.metrics.r2_score(targets, predictions)
        return r2_value

    def get_performance_mse(self, predictions, targets):
        """Calculate MSE value on predictions.

        Parameters
        ----------
        predictions : pd.DataFrame or np.array
            Two-dimensional data structure containing predictions
        targets : pd.DataFrame or np.array
            Two-dimensional data structure containing targets

        Returns
        -------
        float
            Performance MSE value
        """
        mse_value = sklearn.metrics.mean_squared_error(targets, predictions)
        return mse_value


"""
Below is some example code that could be used for importing and testing.
Note that we're measuring performance on the whole dataset here, and on all
targets simultaneously.
"""

if __name__ == '__main__':
    prediction_data = pd.read_csv('data/operation_points_v2.csv')

    predictor = Predictor(prediction_data,
                          path_to_ann_model='models/',
                          path_to_gbr_model='models/')

    features = prediction_data.loc[:, ['Ms', 'Ts_in', 'Mf', 'Tc_in', 'Tc_out']]
    targets = prediction_data.loc[:, ['Md', 'Ts_out', 'Mc']]

    prediction_ann = predictor.predict_ann(prediction_data)
    prediction_gbr = predictor.predict_gbr(prediction_data)

    performance_r2_ann = predictor.get_performance_r2(
        prediction_ann, targets)
    performance_r2_gbr = predictor.get_performance_r2(
        prediction_gbr, targets)
    performance_mse_ann = predictor.get_performance_mse(
        prediction_ann, targets)
    performance_mse_gbr = predictor.get_performance_mse(
        prediction_gbr, targets)

    print(performance_mse_ann)
    print(performance_r2_ann)
    print(performance_mse_gbr)
    print(performance_r2_gbr)
