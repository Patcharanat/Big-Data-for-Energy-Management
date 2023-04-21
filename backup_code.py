## Imputation code

imputer = IterativeImputer(estimator=BayesianRidge(), # default estimator 
                        max_iter=25, 
                        random_state=1)

data_zone_list = []
for e in zone:
    array_temp = imputer.fit_transform(df[zone[e]].drop(columns=['Date']))
    df_temp = form_dataframe(data_array=array_temp, old_dataframe=df[zone[e]])
    data_zone_list.append(df_temp.drop(columns=['Date']))
df_sample = pd.concat(data_zone_list, axis=1)
df_sample.insert(loc=0, column='Date', value=df['Date'])

df_sample = process_data(df_sample)

# test 5 imputation method

imputer = IterativeImputer(estimator=BayesianRidge(), # default estimator 
                        max_iter=50, 
                        random_state=1,
                        verbose=1)
data_zone_list = []
for e in zone:
    array_temp = imputer.fit_transform(df[zone[e]].drop(columns=['Date']))
    df_temp = ee.form_dataframe(data_array=array_temp, old_dataframe=df[zone[e]])
    data_zone_list.append(df_temp.drop(columns=['Date']))
df_sample1 = pd.concat(data_zone_list, axis=1)
df_sample1.insert(loc=0, column='Date', value=df['Date'])

df_sample1 = ee.process_data(df_sample1)

df_sample2 = imputer_knn_transformer.fit_transform(df)
df_sample3 = imputer_meangroupby_transformer.fit_transform(df)

df_sample4 = imputer_nan.impute(df)
df_sample4 = ee.process_data(df_sample4)

df_sample5 = imputer_nanlegacy.impute(df)
df_sample5 = ee.process_data(df_sample5)

X1, y1 = ee.sep_input_output(df_sample1)
X1_train, X1_test, y1_train, y1_test = ee.train_test_split_data(X1, y1, train_size=0.8)

X2, y2 = ee.sep_input_output(df_sample2)
X2_train, X2_test, y2_train, y2_test = ee.train_test_split_data(X2, y2, train_size=0.8)

X3, y3 = ee.sep_input_output(df_sample3)
X3_train, X3_test, y3_train, y3_test = ee.train_test_split_data(X3, y3, train_size=0.8)

X4, y4 = ee.sep_input_output(df_sample4)
X4_train, X4_test, y4_train, y4_test = ee.train_test_split_data(X4, y4, train_size=0.8)

scaler = StandardScaler()
scaler_X = copy.deepcopy(scaler)
scaler_y1 = copy.deepcopy(scaler)
scaler_y2 = copy.deepcopy(scaler)
scaler_y3 = copy.deepcopy(scaler)
scaler_y4 = copy.deepcopy(scaler)
scaler_y5 = copy.deepcopy(scaler)

X1_train = scaler_X.fit_transform(X1_train)
X1_test = scaler_X.transform(X1_test)
y1_train = scaler_y1.fit_transform(y1_train)

X2_train = scaler_X.fit_transform(X2_train)
X2_test = scaler_X.transform(X2_test)
y2_train = scaler_y2.fit_transform(y2_train)

X3_train = scaler_X.fit_transform(X3_train)
X3_test = scaler_X.transform(X3_test)
y3_train = scaler_y3.fit_transform(y3_train)

X4_train = scaler_X.fit_transform(X4_train)
X4_test = scaler_X.transform(X4_test)
y4_train = scaler_y4.fit_transform(y4_train)

data = [
    (X1_train, y1_train, X1_test, y1_test, scaler_y1),
    (X2_train, y2_train, X2_test, y2_test, scaler_y2),
    (X3_train, y3_train, X3_test, y3_test, scaler_y3),
    (X4_train, y4_train, X4_test, y4_test, scaler_y4),
]

model = DecisionTreeRegressor()
score_temp = ee.init_score_df()
causals = ['IterativeImputer', 'KNNImputer', 'Groupby and Mean', 'LightGBM (NaNImputer)', 'XGBoost (NaNImputerLegacy)']

for i, e in enumerate(data):
    print(i)
    model.fit(e[0], e[1])
    temp_pred = model.predict(e[2])
    score_temp = ee.insert_score(score_temp, 
                                ee.make_predict_score(e[3], 
                                                    e[4].inverse_transform(temp_pred)), 
                                causals[i])
    
# Iterative Imputer test

imputer = IterativeImputer(estimator=BayesianRidge(), # default estimator 
                        max_iter=50,
                        initial_strategy='median' ,
                        random_state=1,)
print('pre-processing')
data_zone_list = []
for e in zone:
    array_temp = imputer.fit_transform(df[zone[e]].drop(columns=['Date']))
    df_temp = ee.form_dataframe(data_array=array_temp, old_dataframe=df[zone[e]])
    data_zone_list.append(df_temp.drop(columns=['Date']))
    print(imputer.n_iter_)
df_sample1 = pd.concat(data_zone_list, axis=1)
df_sample1.insert(loc=0, column='Date', value=df['Date'])

df_sample1 = ee.process_data(df_sample1)

X1, y1 = ee.sep_input_output(df_sample1)
X1_train, X1_test, y1_train, y1_test = ee.train_test_split_data(X1, y1, train_size=0.8)

X1_train = scaler_X.fit_transform(X1_train)
X1_test = scaler_X.transform(X1_test)
y1_train = scaler_y1.fit_transform(y1_train)


print('training model')
model = XGBRegressor()
model.fit(X1_train, y1_train)
temp_pred = model.predict(X1_test)
temp_score = ee.make_predict_score(y1_test, scaler_y1.inverse_transform(temp_pred))
print(temp_score)

# KNNImputer Test

imputer = KNNImputer(n_neighbors=2, weights='uniform') # default parameters
# imputer = KNNImputer(n_neighbors=5, weights='distance')

print('pre-processing')
data_zone_list = []
for e in zone:
    df_temp1 = ee.set_date_as_index(df[zone[e]])
    df_temp2 = ee.create_timeseries_features(df_temp1)
    df_temp2 = ee.set_index_as_index(df_temp2)
    
    array_temp = imputer.fit_transform(df_temp2.drop(columns=['Date']))
    df_temp = ee.form_dataframe(data_array=array_temp, old_dataframe=df_temp2)
    df_temp = df_temp[df_temp1.columns]
    
    data_zone_list.append(df_temp)

df_sample = pd.concat(data_zone_list, axis=1)
df_sample.insert(loc=0, column='Date', value=df['Date'])

df_sample = ee.process_data(df_sample)

X1, y1 = ee.sep_input_output(df_sample)
X1_train, X1_test, y1_train, y1_test = ee.train_test_split_data(X1, y1, train_size=0.8)

scaler = StandardScaler()
scaler_X = copy.deepcopy(scaler)
scaler_y = copy.deepcopy(scaler)

X1_train = scaler_X.fit_transform(X1_train)
X1_test = scaler_X.transform(X1_test)
y1_train = scaler_y.fit_transform(y1_train)


print('training model')
model = XGBRegressor()
model.fit(X1_train, y1_train)
temp_pred = model.predict(X1_test)
temp_score = ee.make_predict_score(y1_test, scaler_y.inverse_transform(temp_pred))

# NaNImputerLegacy

imputer_nanlegacy = NaNImputerLegacy(
    conservative=False,
    n_feats=10,
    nan_cols=None,
    fix_string_nans=True,
    multiprocessing_load=3,
    verbose=True,
    fill_nans_in_pure_text=False,
    drop_empty_cols=True,
    drop_nan_cols_with_constant=True
)
df_sample = imputer_nanlegacy.impute(df)
df_sample = ee.process_data(df_sample)

### cross-validation

## iterate over estimators, cv is also used
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_validate(estimator=temp_model, X=X_train, y=y_train, 
                    cv=tscv, scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'))


## calculate scores for each metric
cv_r2 = scores['test_r2'].mean()
cv_mse = -scores['test_neg_mean_squared_error'].mean()
cv_mae = -scores['test_neg_mean_absolute_error'].mean()
score_each_set = [cv_r2, cv_mse, cv_mae]


## Iterate over Imputation techniques, scaler, ML Models, and CV to find the best combination
# "Iterating approachs and models"
# - Iterate over Imputation techniques
# - Iterate over scaling methods
# - Iterate over ML models
# - Choose the best combination by cross-validation
# Note: criterion or metrics to determine have to change at:
# - scoring in HalvingRandomSearchCV
# - selected metrics at 2 bottoms line of the cell

score_df = ee.init_score_df()
execute_time = {'tune_time': [], 'pred_time': []}
for index_imputer, imputer in enumerate([imputer_iterative_transformer, 
                                         imputer_knn_transformer, 
                                         imputer_meangroupby_transformer]):
    for index_scaler, scaler in enumerate([MinMaxScaler(), 
                                           StandardScaler(), 
                                           RobustScaler()]):
        for index_model, model in enumerate([model_DT, 
                                             model_RF, 
                                             model_svr, 
                                             model_gbm, 
                                             model_adaboost, 
                                             model_xgb, 
                                             model_catboost, 
                                             model_lgbm]):
            # print progress
            print((index_imputer, index_scaler, index_model))
            
            # iterate over imputation methods
            df_sample = imputer.fit_transform(df)
            
            X, y = ee.sep_input_output(df_sample)
            X_train, X_test, y_train, y_test = ee.train_test_split_data(X, y, train_size=0.8)

            # iterate over scalers
            scaler_x = copy.deepcopy(scaler)
            sclaer_y = copy.deepcopy(scaler)
            X_train_scaled = scaler_x.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)
            X_test_scaled = scaler_x.transform(X_test)

            # tune model
            if model == model_DT:
                params = {
                    'criterion': ["squared_error"],
                    'splitter': ['best', 'random'],
                    'max_depth': [i for i in range(2, 10)],
                    'min_samples_leaf': [1, 4, 8, 16, 32],
                    'max_features': ['sqrt', None, 0.2, 0.3, 0.4, 0.5, 0.8],
                    'random_state': [0]
                }
            elif model == model_RF:
                params = {
                    'n_estimators': [100, 500, 1000, 1500],
                    'criterion': ['squared_error'],
                    'min_samples_leaf': [1, 4, 8, 16, 32, 128],
                    'max_features': ['sqrt', None, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9],
                    'bootstrap': [True],
                    'oob_score': [True],
                    'random_state': [0]
                }
            elif model == model_svr:
                pass
            elif model == model_adaboost:
                pass
            elif model == model_gbm:
                pass
            elif model == model_xgb:
                pass
            elif model == model_catboost:
                pass
            elif model == model_lgbm:
                pass
            
            # fitting (tuning parameters)

            tscv = TimeSeriesSplit(n_splits=5)
            search = HalvingRandomSearchCV(estimator=model, param_distributions=params, 
                                        cv=tscv, scoring='mse', random_state=0)
            init_tune_time = time.time()
            search.fit(X_train_scaled, y_train_scaled)
            end_tune_time = time.time() - init_tune_time
            print("Tuning time: ", end_tune_time)
            print(search.best_params_)
            temp_model = search.best_estimator_

            # predict

            init_pred_time = time.time()
            prediction_scaled = temp_model.predict(X_test_scaled)
            end_pred_time = time.time() - init_pred_time
            print("Predict time: ", end_pred_time)
            
            # don't need cross-validation again if it is used in tuning hyperparameters
            
            # inverse scale prediction
            prediction = scaler_y.inverse_transform(prediction_scaled)

            # add score to dataframe score
            causal = str((index_imputer, index_scaler, index_model))
            score_df = ee.insert_score(score_df, ee.make_predict_score(y_test, prediction), causal)
            # also add time
            execute_time['tune_time'].append(end_tune_time)
            execute_time['pred_time'].append(end_pred_time)
            
            break # for testing

score_df_new = pd.concat([score_df, pd.DataFrame(execute_time)], axis=1)
# select metrics important the most
most_score = score_df_new["mse"].max()
print(score_df_new.loc[df["mse"] == most_score])


## neural network
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Define the number of time steps and the number of features
n_steps = 3
n_features = 1

# Create the input and output sequences for the RNN model
X, y = [], []
for i in range(n_steps, len(data_normalized)):
    X.append(data_normalized[i-n_steps:i, 0])
    y.append(data_normalized[i, 0])
X, y = np.array(X), np.array(y)

# Reshape the input data to fit the RNN model
X = np.reshape(X, (X.shape[0], X.shape[1], n_features))

# Define the RNN model architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the RNN model
model.fit(X, y, epochs=50, verbose=0)

# Make predictions on new data
new_data = pd.read_csv('your_new_data_file.csv')
new_data_normalized = scaler.transform(new_data)
new_X = []
for i in range(n_steps, len(new_data_normalized)):
    new_X.append(new_data_normalized[i-n_steps:i, 0])
new_X = np.array(new_X)
new_X = np.reshape(new_X, (new_X.shape[0], new_X.shape[1], n_features))
predictions = model.predict(new_X)

# Inverse transform the predictions to get the original scale
predictions = scaler.inverse_transform(predictions)


## call GBM
# sklearn GBM default parameters
model_gbm = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.1,
                                n_estimators=100,
                                max_depth=3,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                subsample=1))
init_tune_time = time.time()
model_gbm.fit(X_train, y_train)
end_fit_time = time.time() - init_tune_time

init_pred_time = time.time()
prediction_gbm = model_gbm.predict(X_test)
end_pred_time = time.time() - init_pred_time

prediction_gbm_train = model_gbm.predict(X_train)

print("fit time: ", end_fit_time, "\npred time: ", end_pred_time)

# plot early stopping
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

time_list = []
score_r2_list = []
score_mse_list = []

# default parameters
print('processing...model 1')
model_xgb1 = XGBRegressor()
time_start = time.time()
model_xgb1.fit(X_train, y_train)
time_list.append(time.time() - time_start)
score_r2_list.append(r2_score(y_test, model_xgb1.predict(X_test)))
score_mse_list.append(mean_squared_error(y_test, model_xgb1.predict(X_test), squared=False))

# without early stopping n_estimator: 100
print('processing...model 2')
model_xgb2 = XGBRegressor(n_estimators=200)
time_start = time.time()
model_xgb2.fit(X_train, y_train)
time_list.append(time.time() - time_start)
score_r2_list.append(r2_score(y_test, model_xgb2.predict(X_test)))
score_mse_list.append(mean_squared_error(y_test, model_xgb2.predict(X_test), squared=False))

# with early stopping n_estimators: 100
print('processing...model 3')
model_xgb3 = XGBRegressor(n_estimators=200, learning_rate=0.1, early_stopping_rounds=20)
time_start = time.time()
model_xgb3.fit(X_train_new, y_train_new, eval_set=[(X_eval, y_eval)], eval_metric='rmse')
time_list.append(time.time() - time_start)
score_r2_list.append(r2_score(y_test, model_xgb3.predict(X_test)))
score_mse_list.append(mean_squared_error(y_test, model_xgb3.predict(X_test), squared=False))

# use n_estimators from early stopping criterion
print('processing...model 4')
model_xgb4 = XGBRegressor(n_estimators=model_xgb3.best_iteration+1)
time_start = time.time()
model_xgb4.fit(X_train, y_train)
time_list.append(time.time() - time_start)
score_r2_list.append(r2_score(y_test, model_xgb4.predict(X_test)))
score_mse_list.append(mean_squared_error(y_test, model_xgb4.predict(X_test), squared=False))

print('training models finish')

# n_estimators validation_curve fort xgboost
from sklearn.model_selection import validation_curve

# define the number of estimators to evaluate
n_estimators = np.arange(50, 1000, 50)

# compute the validation curve
train_scores, test_scores = validation_curve(XGBRegressor(), X_train, y_train, 
                                             param_name='n_estimators', 
                                             param_range=n_estimators, 
                                             scoring='neg_mean_squared_error', 
                                             cv=tscv)

# compute the mean and standard deviation of the scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# plot the learning curve
plt.plot(n_estimators, train_mean, color='blue', marker='o', markersize=5, label='training error')
plt.fill_between(n_estimators, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(n_estimators, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation error')
plt.fill_between(n_estimators, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper right')
plt.show()

print(n_estimators[list(test_mean).index(min(test_mean))])

# n_estimators validation curve for catboost
from sklearn.model_selection import validation_curve

# define the number of estimators to evaluate
iterations_number = np.arange(100, 550, 50)

# compute the validation curve
train_scores, test_scores = validation_curve(MultiOutputRegressor(CatBoostRegressor(verbose=0)), X_train, y_train, 
                                             param_name='estimator__iterations', 
                                             param_range=iterations_number, 
                                             scoring='neg_mean_squared_error', 
                                             cv=tscv)

# compute the mean and standard deviation of the scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# plot the learning curve
plt.plot(iterations_number, train_mean, color='blue', marker='o', markersize=5, label='training error')
plt.fill_between(iterations_number, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(iterations_number, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation error')
plt.fill_between(iterations_number, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper right')
plt.show()

print(iterations_number[list(test_mean).index(min(test_mean))])

# n_estimators validation curve for lightgbm 
from sklearn.model_selection import validation_curve

# define the number of estimators to evaluate
# iterations_number = np.arange(100, 650, 50)
iterations_number = np.arange(600, 1050, 50)

# compute the validation curve
train_scores, test_scores = validation_curve(MultiOutputRegressor(LGBMRegressor()), X_train, y_train, 
                                             param_name='estimator__n_estimators', 
                                             param_range=iterations_number, 
                                             scoring='neg_mean_squared_error', 
                                             cv=tscv)

# compute the mean and standard deviation of the scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# plot the learning curve
plt.plot(iterations_number, train_mean, color='blue', marker='o', markersize=5, label='training error')
plt.fill_between(iterations_number, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(iterations_number, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation error')
plt.fill_between(iterations_number, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper right')
plt.show()

print(iterations_number[list(test_mean).index(min(test_mean))])