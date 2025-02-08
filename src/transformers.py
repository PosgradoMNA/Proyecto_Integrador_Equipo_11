"""Custom transformers for feature engineering."""

import pandas as pd
import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

class WaveletTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom scikit transformer for wavelet transformation of time-series data.
    '''

    def _extract_wavelet_features(self, data_list, wavelet='db4', level=10):
        """
        Aplicar la Transformada Wavelet a una lista de datos y extraer características.

        Parámetros:
        data_list = lista de datos a usar.
        wavelet = transformacion wavelet a usar (Daubechies 4 wavelet)
        level = número de veces que se realiza el proceso de descomposición en una señal o imagen
        """
        coeffs = pywt.wavedec(data_list, wavelet, level=level)
        features = []
        for coef in coeffs:
            features.extend([
                np.mean(coef),
                np.std(coef),
                np.min(coef),
                np.max(coef)
            ])
        return features

    def fit(self, X, y=None):
        # stateless transformer
        self.column_names = list(X.columns)

        return self

    def transform(self, X, y=None):
        return X.apply(self._batch_process)

    def _apply_transformation(self, window, column_name):
        return self._extract_wavelet_features(window)

    def _batch_process(self, column):
        return column.map(lambda col_val: np.array(self._apply_transformation(col_val, column.name)))

    def get_feature_names_out(self, column_names):
        return self.column_names


class StandardScalerTSTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom scikit scaler transformer for time-series data.
    '''

    def fit(self, X, y=None):
        self.x_mean = X['x'].explode().mean()
        self.x_std = X['x'].explode().std()
        self.y_mean = X['y'].explode().mean()
        self.y_std = X['y'].explode().std()
        self.column_names = list(X.columns)

        return self

    def transform(self, X, y=None):
        normalized_x = (X['x'].to_list() - self.x_mean) / self.x_std
        normalized_y = (X['y'].to_list() - self.y_mean) / self.y_std

        new_x_column = list(pd.DataFrame(normalized_x).values)
        new_y_column = list(pd.DataFrame(normalized_y).values)
        transformed_columns = pd.DataFrame([new_x_column, new_y_column]).T
        transformed_columns.columns = self.column_names

        return transformed_columns

    def get_feature_names_out(self, column_names):
        return self.column_names


class LLETransformer(BaseEstimator, TransformerMixin):
    '''
    Custom scikit transformer LLE transformation of time-series data.
    '''

    def fit(self, X, y=None):
        self.column_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        self.x_train_data = pd.DataFrame(X['x'].to_list())
        self.y_train_data = pd.DataFrame(X['y'].to_list())
        
        x_embedding = LocallyLinearEmbedding(n_components=6, n_neighbors=4, eigen_solver='dense')
        y_embedding = LocallyLinearEmbedding(n_components=6, n_neighbors=4, eigen_solver='dense')

        x_lle_data = x_embedding.fit_transform(self.x_train_data)
        y_lle_data = y_embedding.fit_transform(self.y_train_data)


        new_x_column = list(pd.DataFrame(x_lle_data).values)
        new_y_column = list(pd.DataFrame(x_lle_data).values)
        transformed_columns = pd.DataFrame([new_x_column, new_y_column]).T

        return transformed_columns

    def get_feature_names_out(self, column_names):
        return self.column_names


class PCATransformer(BaseEstimator, TransformerMixin):
    '''
    Custom scikit transformer PCA transformation of time-series data.
    '''

    def fit(self, X, y=None):
        self.column_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        self.x_train_data = pd.DataFrame(X['x'].to_list())
        self.y_train_data = pd.DataFrame(X['y'].to_list())

        x_pca = PCA(n_components=6)
        y_pca = PCA(n_components=6)

        x_pca_data = x_pca.fit_transform(self.x_train_data)
        y_pca_data = y_pca.fit_transform(self.y_train_data)


        new_x_column = list(pd.DataFrame(x_pca_data).values)
        new_y_column = list(pd.DataFrame(y_pca_data).values)
        transformed_columns = pd.DataFrame([new_x_column, new_y_column]).T

        return transformed_columns

    def get_feature_names_out(self, column_names):
        return self.column_names

