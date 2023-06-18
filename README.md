# EE-Project: Big Data for Energy Management
*By Patcharanat P.*

## Introduction
Chulalongkorn University is an educational institute that consumes enormous energy power referring to the [CUBEMS website](https://www.bems.chula.ac.th/web/cham5/#/) monitoring building energy consumption. Hence, it seems to be utilizable and applicable if we are able to implement the data collected from the [CUBEMS project](https://www.nature.com/articles/s41597-020-00582-3) with Machine Learning and Data Improvement to initiate an idea of energy management and future research to optimize energy costs for Chulalongkorn University. Understanding the energy need could be a step further in planning the resources therefore, it became the main idea of this project to be created.

In this project, the author emphasized machine learning model development, result evaluation, and data Improvement using the [CUBEMS dataset](https://github.com/Patcharanat/Big-Data-for-Energy-Management/tree/master/dataset), which recorded the Charmchuri 5 building’s energy consumption and characteristic of the environment. Most of the important works are written and some which are less relevant are only mentioned but not written in detail to give an idea for future works or further research.

***Keywords:** Machine Learning, Imputation, Data manipulation, Data Science, Energy management*

## Project Processes *(What to present in the project)*
- Data Pre-processing and Feature Engineering
    - **Imputation: ItertiveImputer, KNNImputer, NaNImputer**
    - Time-series feature engineering
    - Scaling: MinMaxScaler, StandardScaler, RobustScaler
- [Model Research](https://github.com/Patcharanat/ML-Learning/blob/master/ML_research.md) - (algorithms and hyperparameters)
    - Decision Tree
    - Random Forest
    - SVM: LinearSVR, SVR (poly, rbf)
    - GBM (scikit-learn)
    - HistGBM (scikit-learn)
    - AdaBoost
    - XGBoost
    - CatBoost
    - LightGBM
    - KNN
    - K-Means
- Model Tuning
    - RandomizedSearchCV
    - HalvingRandomSearchCV
    - Cross-validation: TimeSeriesSplit
- Other techniques
    - Early stopping
    - Validation curve
    - `best_iteration` of Tree-based models
- Model Evaluation
    - R2, MSE, MAE
    - Model performance comparison
    - **Feature importance on tree-based model**
    - **Imputation performance comparison**
    - **Visualization of model output**
- Further application
    - Data Transformation (for clustering)
    - **Clustering with K-Means**
## What this repo contains
1. [pre-project-notebook-sample.ipynb](pre-project-notebook-sample.ipynb)
    - Notebook for 1st phase of the project
2. [ee_functions.py](ee_functions.py)
    - Functions script for data pre-processing and imputation
3. [**ee-project-prototype.ipynb**](ee-project-prototype.ipynb)
    - **Main Notebook developing the project**

    *Disclaimer: The notebook contained code and experiments mostly without explanation, and re-running notebook could take more than 6 hours for models tuning*

    ***See [Project Processes](#project-processes-what-to-present-in-the-project) and [Project Summary](#project-summary) to read explanation.*** 
4. [ee-project-eda.ipynb](ee-project-eda.ipynb)
    - Some ad-hoc analysis and visualization 
5. [clustering_zone.ipynb](clustering_zone.ipynb)
    - Further application after *ee-project-prototype.ipynb*
6. [backup_code.py](backup_code.py)
    - backup code that was used in the project
7. Folder [dataset](https://github.com/Patcharanat/Big-Data-for-Energy-Management/tree/master/dataset)
    - Contained original dataset of CUBEMs project
8. Folder [data_sample](https://github.com/Patcharanat/Big-Data-for-Energy-Management/tree/master/data_sample)
    - Contained processed datasets for reproducing experiments
9. Folder [Project Progress](https://github.com/Patcharanat/Big-Data-for-Energy-Management/tree/master/Project%20Progress)
    - Contained experiments result, score and important plot
10. Additional files: `README.md`, `.gitignore`
    - not directly relevant to the project
## Project Summary
Project Overview

<img src="./Project%20Progress/plot/project-overview.jpg" alt="project-overview.png" width="75%">

The Project consists of 3 main parts, including Data Improvement, Model Development, and Model Application.

- Data Improvement
    - How to transform and automated handle existing data to be input to develop models
    - How to impute missing values due to incompleted datasets
- Model Development
    - Data Pre-processing and Feature Engineering
    - Develop models to predict energy consumption
        - Tune hyperparameters with Time series cross-validation
        - Learn algorithms
    - Evaluate model performance (both imputer and regression model)
- Model Application
    - How to further exploit model output to be useful for energy management
    - Transform data and apply clustering to find the pattern of energy consumption

### 1. Raw Data EDA

<img src="./Project%20Progress/plot/raw-eda.jpg" alt="raw-eda" width="75%">

Firstly, using all datasets would be too big to develop ML models because of long training time. Therefore, we need to choose one which has proper characteristics (less outliers) unless we have to perform outliers removal. As showed in the picture, each floor has different characteristics of energy consumption. If we need to use model that trained by one floor to predict other floor, we need to re-train model with that floor. The author chose the 4th floor because it has the least outliers.
### 2. Imputation Techniques
<img src="./Project%20Progress/plot/impute-process-evaluate.jpg" alt="impute-process-evaluate" width="75%">

In statistic, one can evalute how well they impute missing values by fixing a model and the way they pre-process data, then use it as an input to get scores such as R-squared, MSE, and MAE from the prediction made by the fixed model as showed in the picture.

Even there're models that handle missing values such as RandomForest, XGBoost, LightGBM, and CatBoost, but it still not an efficient way to impute the missing values without logic or knowledge, hence it came to imputaion techniques experiment in this project.

<img src="./Project%20Progress/plot/all-impute.jpg" alt="all-impute" width="75%">

The author presented 3 techniques with ML Imputer, including IterativeImputer, KNNImputer, and NaNImputer.

### 3. Feature Engineering

In this project, the author used 2 types of feature engineering, including time-series feature engineering and rolling mean. The reason that the author didn't emphasize this part is to limit scope of the project. However, it's still important to do feature engineering to improve model performance.

order of feature engineering and imputation affect model performance, because using model for imputation is a process of impute  
### 4. Scaling

### 5. Tuning

### 6. Model development techniques

Validation Curve

<img src="./Project%20Progress/plot/n_estimators_validation_curve_xgb.png" alt="n_estimators_validation_curve_xgb" width="75%">




### 7. Model Evaluation
Model performance comparison

<img src="./Project%20Progress/plot/model_final.png" alt="model_final" width="75%">

Imputation performance comparison

<img src="./Project%20Progress/plot/imputation_final.png" alt="imputation_final" width="75%">

Prediction Visualization

<img src="./Project%20Progress/plot/output-lgbm.png" alt="output-lgbm" width="75%">

<img src="./Project%20Progress/plot/output-lgbm-zoom3.png" alt="output-lgbm-zoom3" width="75%">

<img src="./Project%20Progress/plot/output-lgbm-zoom1.png" alt="output-lgbm-zoom1" width="75%">

<img src="./Project%20Progress/plot/output-lgbm-zoom2.png" alt="output-lgbm-zoom2" width="75%">

Effect of imputation on outlier

<img src="./Project%20Progress/plot/impute-outlier-raw.png" alt="impute-outlier-raw" width="75%">

<img src="./Project%20Progress/plot/impute-outlier-impute.png" alt="impute-outlier-impute" width="75%">

### 8. Further application

Clustering
## Conclusion
-

***Writing README.md . . .***

---