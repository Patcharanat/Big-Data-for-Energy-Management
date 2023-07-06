# EE-Project: Big Data for Energy Management
*By Patcharanat P.*

## Introduction
Chulalongkorn University is an educational institute that consumes enormous energy power referring to the [CUBEMS website](https://www.bems.chula.ac.th/web/cham5/#/) monitoring building energy consumption. Hence, it seems to be utilizable and applicable if we are able to implement the data collected from the [CUBEMS project](https://www.nature.com/articles/s41597-020-00582-3) with Machine Learning and Data Improvement to initiate an idea of energy management and future research to optimize energy costs for Chulalongkorn University. Understanding the energy need could be a step further in planning the resources therefore, it became the main idea of this project to be created.

In this project, the author emphasized machine learning model development, result evaluation, and data Improvement using the [CUBEMS dataset](https://github.com/Patcharanat/Big-Data-for-Energy-Management/tree/master/dataset), which recorded the Charmchuri 5 building’s energy consumption and characteristic of the environment. Most of the important works are written and some which are less relevant are only mentioned but not written in detail to give an idea for future works or further research.

***Keywords:** Machine Learning, Imputation, Data manipulation, Data Science, Energy management*

## **Presenting in the Project**

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
- Other Techniques
    - Early stopping
    - Validation curve
    - `best_iteration` of Tree-based models
- Model Evaluation
    - R2, MSE, MAE
    - Model performance comparison
    - **Feature importance on tree-based model**
    - **Imputation performance comparison**
    - **Visualization of model output**
- Further Application
    - Data Transformation (for clustering)
    - **Clustering with K-Means**
## What this repo contains
1. [pre-project-notebook-sample.ipynb](pre-project-notebook-sample.ipynb)
    - Notebook for the first phase of the project
2. [ee_functions.py](ee_functions.py)
    - Functions script for data pre-processing and imputation
3. [**ee-project-prototype.ipynb**](ee-project-prototype.ipynb)
    - **Main Notebook developing the project** (2nd phase)

    *Disclaimer: The notebook contained code and experiments mostly without explanation, and re-running notebook could take more than 6 hours for models tuning*

    ***See [Presenting in the Project](#presenting-in-the-project) and [Project Summary](#project-summary) to read explanation.*** 
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
## Project Summary
**Project Overview**

<img src="./Project%20Progress/plot/project-overview.jpg" alt="project-overview.png" width="75%">

The Project consists of 3 main parts, including Data Improvement, Model Development, and Model Application.

- Data Improvement
    - How to transform and automatic handle existing data to be input feeding to models
    - How to impute missing values using statistic and Machine Learning Imputers due to incompleted datasets
- Model Development
    - Data Pre-processing and Feature Engineering
    - Develop models to predict energy consumption
        - Tune hyperparameters with Time series cross-validation
        - Learn algorithms
    - Evaluate model performance (both imputer and regression model)
- Model Application
    - How to further exploit model output to be useful for energy management
    - Transform data and apply clustering algorithm to find the pattern of energy consumption

list of contents
- [1. Raw Data EDA](#1-raw-data-eda)
- [2. Imputation Techniques](#2-imputation-techniques)
- [3. Feature Engineering](#3-feature-engineering)
- [4. Scaling Data](#4-scaling-data)
- [5. Tuning Hyperparameters](#5-tuning-hyperparameters)
- [6. Model Development Techniques](#6-model-development-techniques)
- [7. Model Evaluation](#7-model-evaluation)
- [8. Further Application](#8-further-application)
- [Conclusion](#conclusion)
---
### 1. Raw Data EDA

<img src="./Project%20Progress/plot/raw-eda.jpg" alt="raw-eda" width="75%">

Firstly, using all datasets would be too big to develop ML models due to long training time. Therefore, we need to choose one which has proper characteristics (less outliers) unless we have to perform outliers removal. As shown in the picture, each floor has different characteristics of energy consumption. If we need to use model that trained by one floor to predict the others, we need to re-train model with that floor(s). The author chose the 4th floor because it has the least outliers.
### 2. Imputation Techniques
<img src="./Project%20Progress/plot/impute-process-evaluate.jpg" alt="impute-process-evaluate" width="75%">

In statistic, one can evalute how well they impute missing values by set a model and the way they pre-process data, then use them as an input to get scores such as R-squared, MSE, and MAE from the prediction made by the the model. The process can be shown in the picture.

Even there're models that handle missing values such as RandomForest, XGBoost, LightGBM, and CatBoost, but it's still not an efficient way to impute the missing values without the appropriate logic or knowledge on that specific data, hence it came to imputaion techniques experiment in this project.

<img src="./Project%20Progress/plot/all-impute.jpg" alt="all-impute" width="75%">

The author presented 3 techniques with Machine Learning Imputers, including IterativeImputer, KNNImputer, and NaNImputer. The result will be shown in the [Model Evaluation](#7-model-evaluation) part.

### 3. Feature Engineering

In this project, the author used 2 types of feature engineering, including time-series feature engineering and rolling mean. The reason that the author didn't emphasize this part is limited scope of work. However, it's still crucial to do feature engineering to improve model performance.

Order of feature engineering and imputation affected model performance, because using model for imputation is a process of imputing missing values using prediction that come from other features, as I experimented with KNNImputer resulting in the best performance when extracting time-series features before imputation. [Score table for reference]("https://github.com/Patcharanat/Big-Data-for-Energy-Management/blob/master/Project%20Progress/compare_score/compare_score2(impute).parquet")
### 4. Scaling Data

<img src="./Project%20Progress/plot/feature-importance-scale-compare.jpg" alt="feature-importance-scale-compare" width="75%">

This experiment prove that scaling data was important for tree-based models, tested by XGBoostRegressor. The reason is that tree-based models use gradient descent to find the best split, and it's sensitive to the scale of data.

Moreover, scaling should be done separately between input and output if there's relationship between input and output, for example, if output come from mathematical operation of input. And scaling should be done separately from train set and test set to avoid data leakage that can lead to bias increasing. 
### 5. Tuning Hyperparameters

In this project, the author used HalvingRandomizedSearchCV as a tuner to reduce training time, since the accuracy of the model is not the main focus of the project, but instead the training time. The author also used TimeSeriesSplit as a cross-validation technique to reduce overfitting.
### 6. Model Development techniques

Validation Curve

<img src="./Project%20Progress/plot/n_estimators_validation_curve_xgb.png" alt="n_estimators_validation_curve_xgb" width="75%">

As shown in the picture, set fixed hyperparameters, then increasing n_estimators couldn't give one a sense of selecting proper range of n_estimators. Therefore, it's more proper to define search space used in Random search.

Early Stopping

<img src="./Project%20Progress/plot/early-stopping-evaluate.jpg" alt="early-stopping-evaluate" width="75%">

Early Stopping can be used to prevent overfitting and significantly reduce training time. However, the appropriate to use early stopping with TimeSeriesSplit for time series dataset was not covered and discussed in this project.

### 7. Model Evaluation
**Model performance comparison**

<img src="./Project%20Progress/plot/model_final.png" alt="model_final" width="75%">

*Notice: Fitting time was re-training time in second not tuning time*

The best model was LightGBM regarding to the R-squared score, and Mean Squared Error (MSE) score which was more concerned by outliers existence, and Mean Absolute Error (MAE) is not sensitive to outliers.

The tuned model was not better than the default model, because the search space was not large enough to find the best hyperparameters limited by large dataset. However, the tuned model is still significantly better than the default model in term of re-training time.

The tuned random forest model was great approximately equal to default LightGBM model and default CatBoost model, but it took much longer time to train which made it not chosen.

The only linear model, linear_svr, performed poorly because the data was not significantly linearly separable that can be proved by pearson correlation coefficient heatmap below.

<img src="./Project%20Progress/plot/correlation-pearson.png" alt="correlation-pearson" width="75%">

**Imputation performance comparison**

<img src="./Project%20Progress/plot/imputation_final.png" alt="imputation_final" width="75%">

As shown in the picture, KNNImputer outperformed other imputation techniques in term of R-squared score, and Mean Squared Error (MSE) but take much long time to process because it used distance calculation in the algorithm. 

The second best was IterativeImputer which proper to be used in large dataset due to its fast processing time. The worst was NaNImputer considered by fixed pre-processing process and iterating Machine Learning models. NaNImputer resulted in approximately equal R-square score and MSE after feeding to DecisionTree and XGBoost which made it not having been concluded the good performance of NaNImputer.

*[Score by Imputers on Decision Tree](https://github.com/Patcharanat/Big-Data-for-Energy-Management/blob/master/Project%20Progress/compare_score/compare_score6(final_imputation_1floor_decisionTree).parquet), [Score by Imputers on XGBoost](https://github.com/Patcharanat/Big-Data-for-Energy-Management/blob/master/Project%20Progress/compare_score/compare_score6(final_imputation_1floor_xgboost).parquet) as respective references.*

The statistical methods, such as groupby, mean, and median can be considered good methods if simplicity is more concerned than accuracy.

Prediction Visualization

<img src="./Project%20Progress/plot/output-lgbm.png" alt="output-lgbm" width="75%">

<img src="./Project%20Progress/plot/output-zoom.jpg" alt="output-zoom" width="75%">

The default LightGBM model could predict the output quite well, but a bit overfitting which might be the reason of seasonality of the data. the model also able to capture peak load for each day.

Data Interval Effect

<img src="./Project%20Progress/plot/zoom-interval.jpg" alt="zoom-interval" width="75%">

The data interval also affected the model prediction to capture peak load for each day. The lesser interval made it the better prediction, but also led to much longer training time and a chance of overfitting.

The 15-minute interval is acceptable result in term of prediction score and training time.

Effect of imputation on outlier

<img src="./Project%20Progress/plot/outlier-effect.jpg" alt="outlier-effect" width="75%">

The imputation can increase effect of outlier on model prediction, but it's not always the case. The best approach is to remove outlier before using Machine Learning Imputer (or even statistical methods).

### 8. Further application

Clustering

<img src="./Project%20Progress/plot/data-transform-clustering.jpg" alt="data-transform-clustering" width="75%">

After developing the model, the author tried to demonstrae the further usage of model's prediction by clustering the similar energy consumption patterns which might be useful for energy management to know which group of floors should be provided more energy and which floors should be less concerned.

The process was to transform the 33 large datasets into one dataset that represent the energy consumption pattern for each floor, then use K-Means clustering to cluster the data as can be shown above.

<img src="./Project%20Progress/plot/elbow-k-means.jpg" alt="elbow-k-means" width="75%">

As you may familiar with, the elbow method was used to find the best number of clusters. The result was 7 clusters for illustration.

<img src="./Project%20Progress/plot/clustered-eda.png" alt="clustered-eda" width="75%">

After clustering, the author used the same EDA technique to find the pattern of each cluster. The result was as shown above, not only the amount of energy consumption, but also the pattern of energy consumption can be found different from each cluster.

## **Conclusion**

The author has developed the machine learning models that can predict the energy consumption of each floor in the building with acceptable accuracy and training time, and also experimented with the different imputation techniques and pre-processing approaches to find the best processing way for the dataset. The prediction can also be used further to cluster the similar energy consumption patterns which might be useful for us knowing the future energy consumption.

The project gave the author a deep understanding of the machine learning model development process, such as cross-validation, hyperparameter tuning, models' algorithm, mostly tree-based models, model evaluation and other techniques such as early stopping, imputation, data pre-processing, and feature engineering.

*"Because understanding the energy needs, could be a step further in planning the resources properly."*

---