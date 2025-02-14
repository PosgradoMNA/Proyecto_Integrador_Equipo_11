def load_data():
    """
    Carga los datos preprocesados del análisis exploratorio

    Usage:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    """
    X_train = pd.read_parquet('../data/preprocessed/X_train.parquet')
    y_train = pd.read_parquet('../data/preprocessed/y_train.parquet')
    X_val = pd.read_parquet('../data/preprocessed/X_val.parquet')
    y_val = pd.read_parquet('../data/preprocessed/y_val.parquet')
    X_test = pd.read_parquet('../data/preprocessed/X_test.parquet')
    y_test = pd.read_parquet('../data/preprocessed/y_test.parquet')


    assert X_train.shape[0] == y_train.shape[0], "Dimensions don't match"
    assert X_val.shape[0] == y_val.shape[0], "Dimensions, don't match"

    features_x_train = pd.DataFrame(X_train['caracteristica_0'].to_list())
    features_y_train = pd.DataFrame(X_train['caracteristica_1'].to_list())
    all_features_train = pd.concat([features_x_train, features_y_train], axis=1)
    all_features_train.columns = [f"feature_{i}" for i in range(all_features_train.shape[1])]

    features_x_val = pd.DataFrame(X_val['caracteristica_0'].to_list())
    features_y_val = pd.DataFrame(X_val['caracteristica_1'].to_list())
    all_features_val = pd.concat([features_x_val, features_y_val], axis=1)
    all_features_val.columns = [f"feature_{i}" for i in range(all_features_val.shape[1])]

    return all_features_train, y_train, all_features_val, y_val, X_test, y_test



def print_classification_report(y_true, y_pred, label="Depression"):
    print(f"{label} Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"{label} ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")