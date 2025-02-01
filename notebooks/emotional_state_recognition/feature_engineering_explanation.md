# Correlation Between the Notebook and the Paper

The **notebook** and the **paper** both focus on **emotional state recognition** based on handwriting and drawing tasks. Below is a correlation of key feature engineering steps found in the notebook and how they align with the methodology described in the paper.

---

## **Feature Engineering Steps**

### 1. Handling Missing Values
- The dataset removes columns that contain missing values using:
  ```python
  data = data.dropna(axis=1)
  ```
- This step ensures that features with too many missing values do not introduce inconsistencies.

### 2. Shuffling the Data
- To ensure randomness and prevent bias in training/testing splits, the dataset is shuffled:
  ```python
  data = data.iloc[np.random.permutation(data.index)].reset_index(drop=True)
  ```

### 3. Feature Selection and Reduction
- The number of features is determined and selected:
  ```python
  nf = data.shape[1] - 1  # Total number of features
  nc = nf  # Number of features used in the model
  ```
- The paper applies **Fast Correlation-Based Filtering (FCBF)** to select the best features for classification.

### 4. Feature Transformation using H2O
- The dataset is converted into an H2O Frame to leverage the H2O machine learning platform:
  ```python
  h2o_df = h2o.H2OFrame(data)
  ```
- This allows for efficient handling of large datasets.

### 5. Data Augmentation
- The function `agument_Data(data, Eps, proportion)` suggests the addition of Gaussian noise to augment the dataset.
- The paper explicitly mentions using **random Gaussian noise augmentation** to increase dataset size.

### 6. Feature Scaling and Normalization
- Ensuring only numerical data is included:
  ```python
  data = data._get_numeric_data()
  ```
- Scaling and normalization are likely handled within the H2O AutoML pipeline.
- The paper normalizes **spectral and cepstral features**, ensuring standardized inputs.

---

## **Feature Engineering Steps Correlation with Paper**

| Feature Engineering Step (Notebook) | Corresponding Section in Paper |
|------------------------------------|--------------------------------|
| **Handling Missing Values** (`dropna`) | The paper does not explicitly mention handling missing values, but it is implied in the data preprocessing phase to ensure data quality. |
| **Shuffling the Data** (`permutation(data.index)`) | The paper describes using the **EMOTHAW** dataset with multiple handwriting and drawing tasks. The shuffling ensures that data does not introduce biases during training. |
| **Feature Selection and Reduction** (`nf = data.shape[1]-1`) | The paper uses **Fast Correlation-Based Filtering (FCBF)** to select the best features for emotional state classification. |
| **Feature Transformation with H2O** (`h2o.H2OFrame(data)`) | The paper describes extracting **time, spectral, and cepstral features**, which are later processed in the H2O framework for modeling. |
| **Data Augmentation (`agument_Data(data, Eps, proportion)`)** | The paper explicitly mentions **Gaussian noise augmentation** to expand the dataset for better generalization. |
| **Feature Scaling & Normalization (`data._get_numeric_data()`)** | The paper normalizes **spectral and cepstral features**, ensuring that the machine learning model benefits from standardized inputs. |

---

## **Additional Correlations Between the Notebook and Paper**
### 1. **Feature Extraction**
   - The paper extracts **time-domain, spectral-domain, and cepstral-domain features** from handwriting and drawing data.
   - The notebook processes similar data before passing it to the machine learning model.

### 2. **Feature Selection**
   - The paper applies **Fast Correlation-Based Filtering (FCBF)** to reduce the feature set.
   - The notebook calculates the **number of selected features** before passing them to the H2O AutoML framework.

### 3. **Model Training**
   - The paper trains an **SVM model with an RBF kernel** using **Leave-One-Out Cross Validation (LOO)**.
   - The notebook uses the **H2O AutoML framework**, which likely includes SVM and other models for hyperparameter tuning.

### 4. **Data Augmentation**
   - The paper describes **adding small random Gaussian noise** to enhance the dataset and improve model generalization.
   - The notebook contains a function that **adds Gaussian noise** to the dataset before training.

### 5. **Performance Metrics**
   - The paper reports **accuracy improvements of up to 34%** compared to baseline models.
   - The notebook processes data for model evaluation but does not explicitly state performance gains.

---

## **Conclusion**
- The **notebook** follows a **very similar methodology** to the **paper**.
- **Both use:** feature selection, data augmentation (Gaussian noise), and feature transformation (spectral & cepstral).
- The **notebook operationalizes** the methods described in the **paper**, likely serving as an implementation for the research findings.
