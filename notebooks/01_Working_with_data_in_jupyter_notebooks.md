---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: pytorch
    language: python
    name: python3
---

# Working with data in Jupyter notebooks

### Predictive modelling with machine learning

#### Lecturer: Vegard H. Larsen


## Introduction 

### What we will cover in this notebook:
1. Setting up the environment
2. Reading and Writing Data with Pandas
3. Exploring and Visualizing Data
4. Handling Missing Values and Outliers
5. Encoding Categorical Variables
6. Feature Scaling and Normalization
7. Train-Test Split and Basic Data Pipelines
8. Introduction to PyTorch Tensors


## 2. Setting up the environment

Using a virtual environment or a conda environment is crucial for ensuring reproducibility, consistency, and maintainability in your projects. Without an isolated environment, library installations and updates can affect your system-wide settings or other projects, leading to version conflicts and unpredictable behavior. By creating a dedicated environment for each project, you can precisely control which versions of Python and its packages are used, making it easier to replicate your results, share your work with others, and quickly recover a working setup if something goes wrong. This practice streamlines collaboration, simplifies troubleshooting, and ultimately helps maintain the integrity and reliability of your codebase.

- The core libraries we will be using in this course are:
    - pandas for data manipulation and exploration.
        - [Docs](https://pandas.pydata.org/docs/)
    - scikit-learn for preprocessing and modeling
        - [Docs](https://scikit-learn.org/stable/)
    - matplotlib for basic plotting and data visualization
        - [Docs](https://matplotlib.org/stable/contents.html)
    - seaborn for statistical data visualization
        - [Docs](https://seaborn.pydata.org/)
- Additional libraries that will be used/discussed:
    - PyTorch for deep learning workflows (optional)
        - PyTorch is not a core library, but it is widely used for deep learning tasks. We will only touch on it briefly in this course, but you may want to explore it further if you are interested in deep learning. 
        - [Docs](https://pytorch.org/docs/stable/index.html)

```python
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import sklearn
```

```python
# Print our the versions of the libraries
print(f'Pandas version: {pd.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'Matplotlib version: {plt.matplotlib.__version__}')
print(f'Seaborn version: {sns.__version__}')
print(f'Scikit-learn version: {pd.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'Sklearn version: {sklearn.__version__}')
# Check if GPU is available
print(f'Acess to GPU: {torch.cuda.is_available()}') 
```

## 2. Reading and Writing Data with Pandas
- **Read data** from common file formats such as CSV.
- **Write processed data** back to disk in CSV format.
- Use **basic inspection methods** (such as `head()`, `info()`, and `describe()`) to quickly understand the structure and statistical properties of your dataset.

```python
# Reading data
df = pd.read_csv('../data/house-prices/test.csv')  

# Displaying the first 5 rows
df.head()
```

```python
# .info() method to get a summary of the dataframe 

df.info()
```

```python
# .describe() method to get a statistical summary of the dataframe

df.describe()
```

```python
# Writing data to a csv file

df.to_csv('../data/tmp/processed_housing.csv', index=False)
```

## 3. Exploring and Visualizing Data
- Basic summary statistics.
- Identifying distributions of features.
- Simple visualizations (histograms, box plots, scatter plots) to understand data distribution, outliers, and relationships between variables.

```python
# Load an example dataset
tips = sns.load_dataset('tips')

# Quick summary statistics
print(tips.describe())

# Pairplot to visualize relationships
print("Note: pairplot can be slow for larger datasets.\n")

sns.pairplot(tips, hue='time')
plt.show()

# Let's also do a boxplot for 'total_bill' grouped by 'day'\n",
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()
```

## 4. Handling Missing Values and Outliers
- Techniques for detecting missing values (`isnull().sum()`) and outliers (using IQR or $z$-score).
- Strategies for handling missing data (drop vs. impute).
- Using `sklearn.impute.SimpleImputer` for numerical and categorical data.
- Discussion of domain knowledge in deciding how to handle anomalies.

```python
from sklearn.datasets import load_diabetes

# Load the diabetes dataset as a DataFrame
data = load_diabetes(as_frame=True)
df_missing = data.frame

# Artificially introduce some missing values\n",
df_missing.iloc[:10, 2] = np.nan  # Suppose the 3rd column has missing for first 10 rows
  
# Detect missing values
print("Missing values per column:\n", df_missing.isnull().sum())

# Simple strategy: fill numerical missing values with the mean
df_missing.fillna(df_missing.mean(), inplace=True)

# Detect outliers in 'bmi' column using IQR
Q1 = df_missing['bmi'].quantile(0.25)
Q3 = df_missing['bmi'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_missing[(df_missing['bmi'] < Q1 - 1.5 * IQR) | (df_missing['bmi'] > Q3 + 1.5 * IQR)]
print(f"Number of outliers in 'bmi': {len(outliers)}")
```

## 5. Encoding Categorical Variables
- Importance of converting string labels into numeric form for modeling.
- One-hot encoding with `pd.get_dummies()` or `sklearn.preprocessing.OneHotEncoder`.
- Label encoding vs. one-hot encoding and when to use each.

```python
import seaborn as sns

# Load the 'tips' dataset, which has some categorical features
tips = sns.load_dataset('tips')
print("Data types before encoding:\n", tips.dtypes)
print("\nData before encoding:\n", tips.head())

# One-hot encoding on categorical columns
encoded_tips = pd.get_dummies(tips, columns=['day','sex','smoker','time'], drop_first=True)

print("\nData after one-hot encoding:\n", encoded_tips.head())
```

## 6. Feature Scaling and Normalization
- Show `StandardScaler` and `MinMaxScaler` from `scikit-learn`.
- Discuss when scaling is necessary (e.g., for neural networks or distance-based models).

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Iris dataset as a DataFrame
iris = load_iris(as_frame=True)
df_iris = iris.data.copy()

# Standard scaling
scaler = StandardScaler()
df_standard_scaled = scaler.fit_transform(df_iris)
 
# Min-Max scaling
minmax = MinMaxScaler()
df_minmax_scaled = minmax.fit_transform(df_iris)

print("Original (first 5 rows):\n", df_iris.head(), "\n")
print("Standard Scaled (first 5 rows):\n", df_standard_scaled[:5], "\n")
print("Min-Max Scaled (first 5 rows):\n", df_minmax_scaled[:5])
```

The difference between `.fit_transform()`, `.fit()`, and `.transform()`:
1.	`.fit()`:
    - Calculates the parameters required for transformation based on the input data.
    - Does not return transformed data.
    - Example: In `StandardScaler`, `.fit()` calculates the mean and standard deviation.
2.	`.transform()`:
    - Applies the transformation to the data using parameters calculated during `.fit()`.
    - Requires that `.fit()` has been called earlier (either directly or implicitly).
    - Example: In `StandardScaler`, `.transform()` scales the data using the precomputed mean and standard deviation.
3.	`.fit_transform()`:
    - Combines `.fit()` and `.transform()` in one step.
    - Useful when you need to fit and transform the same dataset in a single line.
    - Example: In `StandardScaler`, `.fit_transform()` computes the mean and standard deviation (fit) and then scales the data (transform).


## 7. Train-Test Split and Basic Data Pipelines
- Introduce the concept of splitting data into training and test sets.
- Show `train_test_split` usage from `scikit-learn`.
- Introduce basic pipeline concepts (`sklearn.pipeline.Pipeline`) to ensure consistent preprocessing and modeling steps.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load iris dataset for demonstration
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Pipeline test accuracy: {score:.2f}")
```

## 8. Introduction to PyTorch Tensors
- Briefly show how PyTorch tensors differ from NumPy arrays and how to convert between them.
- This will be relevant for deep learning sessions later in the course.

```python
# Create a simple NumPy array
np_array = np.array([[1, 2], [3, 4]])
print("NumPy Array:")
print(np_array)

# Convert NumPy array to PyTorch tensor
torch_tensor = torch.tensor(np_array)
print("\nPyTorch Tensor:")
print(torch_tensor)

# Perform operations
# Element-wise addition
np_result = np_array + 2
torch_result = torch_tensor + 2

print("\nNumPy Array after adding 2:")
print(np_result)

print("\nPyTorch Tensor after adding 2:")
print(torch_result)

# Key differences:
# 1. NumPy arrays are part of the NumPy library and are used for general-purpose numerical computations.
# 2. PyTorch tensors are similar to NumPy arrays but support GPU acceleration and are optimized for deep learning.

# Example: Moving a PyTorch tensor to a GPU (if available)
if torch.cuda.is_available():
    torch_tensor_gpu = torch_tensor.to('cuda')
    print("\nPyTorch Tensor moved to GPU:")
    print(torch_tensor_gpu)
else:
    print("\nGPU is not available, tensor remains on CPU.")
```


