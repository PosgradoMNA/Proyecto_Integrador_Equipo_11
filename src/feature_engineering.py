import os
# Bibliotecas para manejo de datos
import numpy as np
import pandas as pd

# Bibliotecas para preprocesamiento y transformación
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Bibliotecas para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Importar clases de transformadores custom
from transformers import WaveletTransformer, LLETransformer, PCATransformer, StandardScalerTSTransformer

# parameters for normalization
SCALE = 0.01
PROPORTION = 0.2

np.random.seed(42)

def load_data(raw_data_dir):
    """
    Carga los datos preprocesados del análisis exploratorio
    """
    X_train = pd.read_parquet(os.path.join(raw_data_dir, 'X_train.parquet'))
    y_train = pd.read_parquet(os.path.join(raw_data_dir, 'y_train.parquet'))
    X_val = pd.read_parquet(os.path.join(raw_data_dir, 'X_val.parquet'))
    y_val = pd.read_parquet(os.path.join(raw_data_dir, 'y_val.parquet'))
    X_test = pd.read_parquet(os.path.join(raw_data_dir, 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join(raw_data_dir, 'y_test.parquet'))
    return X_train, y_train, X_val, y_val, X_test, y_test


def data_normalization(X_train, y_train, X_val, y_val, X_test, y_test):
    """Normaliza los datos de entrada"""
    x_mean = X_train['x'].explode().mean()
    x_std = X_train['x'].explode().std()
    y_mean = X_train['y'].explode().mean()
    y_std = X_train['y'].explode().std()

    normalized_x = (X_train['x'] - x_mean) / x_std
    normalized_y = (X_train['y'] - y_mean) / y_std

    return pd.concat([X_train[['homework','pen_status']], normalized_x, normalized_y], axis=1)

def augment_data(features_df, labels_df):
    qty_rows_to_augment = int(PROPORTION * features_df.shape[0])
    df_augmented = features_df.copy()
    df_augmented_labels = labels_df.copy()

    for _ in range(qty_rows_to_augment):  # Create new N data
        target_row = np.random.randint(0, features_df.shape[0] - 1)  # Randomly select a row
        original_x_y_arrays = features_df.iloc[target_row][['x','y']].values

        # Generate Gaussian noise with the same shape as the selected row
        random_noise_x = np.random.normal(0, SCALE, size=original_x_y_arrays[0].shape)
        random_noise_y = np.random.normal(0, SCALE, size=original_x_y_arrays[1].shape)
        new_x_data = original_x_y_arrays[0].shape + random_noise_x  # Add noise
        new_y_data = original_x_y_arrays[1].shape + random_noise_y  # Add noise

        # Convert to DataFrame and maintain original structure
        augmented_row = {
            'x': new_x_data,
            'y': new_y_data,
            'homework': features_df.iloc[target_row]['homework'],
            'pen_status': features_df.iloc[target_row]['pen_status'],
        }
        augmented_row = pd.DataFrame([augmented_row])

        augmented_row_label = pd.DataFrame([labels_df.iloc[target_row]])

        # Concatenate the augmented row to the original dataset
        df_augmented = pd.concat([df_augmented, augmented_row], ignore_index=True, axis=0)
        df_augmented_labels = pd.concat([df_augmented_labels, augmented_row_label], ignore_index=True, axis=0)

    return df_augmented, df_augmented_labels

def create_preprocessing_pipeline(reduction_method='lle'):
    columnas_numericas = ['x', 'y']
    preprocessing_pipeline = None
    if reduction_method == 'lle':
        preprocessing_pipeline = Pipeline([
            ('wavelet', WaveletTransformer()),
            ('lle', LLETransformer())
        ])
    elif reduction_method == 'pca':
        preprocessing_pipeline = Pipeline([
            ('wavelet', WaveletTransformer()),
            ('norm', StandardScalerTSTransformer()),
            ('pca', PCATransformer())
        ])

    preprocesador = ColumnTransformer(
        transformers=[
            ('numericas', preprocessing_pipeline, columnas_numericas)
        ],
        remainder='passthrough'
    )
    return preprocesador

def main():
    print("Cargando datos...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(raw_data_dir='../data/raw_binary')
    normalized_train_df = data_normalization(X_train, y_train, X_val, y_val, X_test, y_test)

    # Aumentar datos
    print("Normalizando datos...")
    df_augmented, y_train_augmented = augment_data(normalized_train_df, y_train)

    # Crear y ajustar pipeline
    print("Creando pipeline...")
    pipeline = create_preprocessing_pipeline(reduction_method='lle')
    X_train_preprocessed = pipeline.fit_transform(df_augmented)
    X_val_preprocessed = pipeline.transform(X_val)
    X_test_preprocessed = pipeline.transform(X_test)

    # Convertir a DataFrame para mejor manejo
    columns = [f'caracteristica_{i}' for i in range(X_train_preprocessed.shape[1])]
    X_train_transformed = pd.DataFrame(X_train_preprocessed, columns=columns)
    X_val_transformed = pd.DataFrame(X_val_preprocessed, columns=columns)
    X_test_transformed = pd.DataFrame(X_test_preprocessed, columns=columns)

    # Mostrar dimensiones y primeras filas
    print("Dimensiones después de la transformación:")
    print(f"Train: {X_train_transformed.shape}")
    print(f"Validación: {X_val_transformed.shape}")
    print(f"Test: {X_test_transformed.shape}")

    print("Guardando datos preprocesados...")
    X_train_transformed.to_parquet('../data/preprocessed/X_train.parquet')
    y_train_augmented.to_parquet('../data/preprocessed/y_train.parquet')

    X_val_transformed.to_parquet('../data/preprocessed/X_val.parquet')
    y_val.to_parquet('../data/preprocessed/y_val.parquet')

    X_test_transformed.to_parquet('../data/preprocessed/X_test.parquet')
    y_test.to_parquet('../data/preprocessed/y_test.parquet')


if __name__ == '__main__':
    main()