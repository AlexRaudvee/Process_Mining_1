import pickle
import joblib
import os 

import pandas as pd
import numpy as np

from numpy import mean, std
from joblib import dump, load
from train_test_split import train_test_split_custom
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, Normalizer

# HERE YOU HAVE TO CHOOSE BPI_Challenge_2012 or BPI_Challenge_2017
chosed_dataset = 'BPI_Challenge_2017'

# extract the 1-3 lags
df = pd.read_csv(f'data/{chosed_dataset}_naive.csv')

df['concept:name - lag_1'] = df.groupby('case:concept:name')['concept:name'].shift(1)
df['concept:name - lag_2'] = df.groupby('case:concept:name')['concept:name'].shift(2)
df['concept:name - lag_3'] = df.groupby('case:concept:name')['concept:name'].shift(3)

# define target
df['next concept:name'] = df.groupby('case:concept:name')['concept:name'].shift(-1)

# # Split the DataFrame 
# df = df.iloc[:1000]

# Prepare data

df_train, df_test = train_test_split_custom(df=df, test_size=0.2, lags=True)

columns = ['concept:name' , 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'next concept:name']
label_encoders = {}
for column in columns:
        label_encoder = LabelEncoder()
        df_test[column] = label_encoder.fit_transform(df_test[column])
        df_train[column] = label_encoder.fit_transform(df_train[column])
        df[column] = label_encoder.fit_transform(df[column])
        label_encoders[column] = label_encoder

X_train = df_train[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3']]
X_test = df_test[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3']]

y_train = df_train[['next concept:name']]
y_test = df_test[['next concept:name']]

print(f"""
    ########################################### RFC INFO ###########################################\n
      inputs: {[col for col in X_test.columns]} \n
      target: {[col for col in y_test.columns]} \n
    ################################################################################################\n
""")

if not os.path.exists('model_weights/random_forest.pkl'):

    rf_clf = RandomForestClassifier(n_jobs=-1)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [21, 22, 23],
        'max_depth': [18, 19, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform nested cross-validation
    inner_cv = KFold(n_splits=5, shuffle=False, random_state=None)  # 5-fold inner cross-validation

    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, scoring='accuracy', cv=inner_cv)
    grid_search.fit(X_train, y_train.values.ravel())

    # Get the best model
    best_model = grid_search.best_estimator_

    # Print the results
    rfc_score = best_model.score(X_test, y_test)

    # Save the best model to a file using pickle
    # Create the folder for model weights
    os.makedirs('model_weights', exist_ok=True)
    model_filename = 'model_weights/random_forest.pkl'

    with open(model_filename, 'wb') as model_file:
        pickle.dump(best_model, model_file)
        
else: 
    with open('model_weights/random_forest.pkl', 'rb') as f:
        best_model = pickle.load(f)

# Make predictions on the dataset for adding new column
df['next concept:name rfc'] = label_encoder.inverse_transform(best_model.predict(df[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3']]))

for column in columns:
        df[column] = label_encoders[column].inverse_transform(df[column])



df.to_csv(f'data/{chosed_dataset}_rfc_xgboost.csv')

df = pd.read_csv(f'data/{chosed_dataset}_rfc_xgboost.csv')

# counting elapsed time
df['elapsed time:timestamp'] = df['time:timestamp diff'].shift(-1) 
df['elapsed time:timestamp'] = pd.to_timedelta(df['elapsed time:timestamp'])
df['elapsed time:timestamp'] = df['elapsed time:timestamp'].apply(lambda x: x.total_seconds()).fillna(-0.01)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
df['CreditScore'] = df['CreditScore'].fillna(0)

# lags for time between events in past 

df['elapsed time:timestamp - lag_1'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(1).fillna(-0.00000001)
df['elapsed time:timestamp - lag_2'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(2).fillna(-0.00000001)
df['elapsed time:timestamp - lag_3'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(3).fillna(-0.00000001)
df['elapsed time:timestamp - lag_4'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(4).fillna(-0.00000001)
df['elapsed time:timestamp - lag_5'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(5).fillna(-0.00000001)
df['elapsed time:timestamp - lag_6'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(6).fillna(-0.00000001)
df['elapsed time:timestamp - lag_7'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(7).fillna(-0.00000001)
df['elapsed time:timestamp - lag_8'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(8).fillna(-0.00000001)
df['elapsed time:timestamp - lag_9'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(9).fillna(-0.00000001)
df['elapsed time:timestamp - lag_10'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(10).fillna(-0.00000001)
df['elapsed time:timestamp - lag_11'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(11).fillna(-0.00000001)
df['elapsed time:timestamp - lag_12'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(12).fillna(-0.00000001)
df['elapsed time:timestamp - lag_13'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(13).fillna(-0.00000001)
df['elapsed time:timestamp - lag_14'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(14).fillna(-0.00000001)
df['elapsed time:timestamp - lag_15'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(15).fillna(-0.00000001)
df['elapsed time:timestamp - lag_16'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(16).fillna(-0.00000001)
df['elapsed time:timestamp - lag_17'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(17).fillna(-0.00000001)
df['elapsed time:timestamp - lag_18'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(18).fillna(-0.00000001)
df['elapsed time:timestamp - lag_19'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(19).fillna(-0.00000001)
df['elapsed time:timestamp - lag_20'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(20).fillna(-0.00000001)
df['elapsed time:timestamp - lag_21'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(21).fillna(-0.00000001)
df['elapsed time:timestamp - lag_22'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(22).fillna(-0.00000001)
df['elapsed time:timestamp - lag_23'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(23).fillna(-0.00000001)
df['elapsed time:timestamp - lag_24'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(24).fillna(-0.00000001)
df['elapsed time:timestamp - lag_25'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(25).fillna(-0.00000001)
df['elapsed time:timestamp - lag_26'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(26).fillna(-0.00000001)
df['elapsed time:timestamp - lag_27'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(27).fillna(-0.00000001)
df['elapsed time:timestamp - lag_28'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(28).fillna(-0.00000001)
df['elapsed time:timestamp - lag_29'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(29).fillna(-0.00000001)
df['elapsed time:timestamp - lag_30'] = df.groupby(by='case:concept:name')['elapsed time:timestamp'].shift(30).fillna(-0.00000001)

# preprocess the columns before fitting
preprocessors = {}
for column in df.columns:
    if column == 'time:timestamp':
        df['year'] = df['time:timestamp'].dt.year
        df['month'] = df['time:timestamp'].dt.month
        df['day'] = df['time:timestamp'].dt.day
        df['hour'] = df['time:timestamp'].dt.hour

    elif column in ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'next concept:name rfc', 'org:resource']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        preprocessors[column] = le

    else:
        continue
    
# split the data on train and test dataframes 
df_train, df_test = train_test_split_custom(df=df, lags=False, test_size=0.2)

# define the input and outputs for the model before put in the train 

X = df[['concept:name',  # this X is for making prediction on the dataframe for saving results in the csv
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'next concept:name rfc', 
                    'year', 
                    'month', 
                    'day', 
                    'hour',
                    'CreditScore',
                    'org:resource',
                    'elapsed time:timestamp - lag_1', 
                    'elapsed time:timestamp - lag_2', 
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10',
                    'elapsed time:timestamp - lag_11',
                    'elapsed time:timestamp - lag_12',
                    'elapsed time:timestamp - lag_13',
                    'elapsed time:timestamp - lag_14',
                    'elapsed time:timestamp - lag_15',
                    'elapsed time:timestamp - lag_16',
                    'elapsed time:timestamp - lag_17',
                    'elapsed time:timestamp - lag_18',
                    'elapsed time:timestamp - lag_19',
                    'elapsed time:timestamp - lag_20',
                    'elapsed time:timestamp - lag_21',
                    'elapsed time:timestamp - lag_22',
                    'elapsed time:timestamp - lag_23',
                    'elapsed time:timestamp - lag_24',
                    'elapsed time:timestamp - lag_25',
                    'elapsed time:timestamp - lag_26',
                    'elapsed time:timestamp - lag_27',
                    'elapsed time:timestamp - lag_28',
                    'elapsed time:timestamp - lag_29',
                    'elapsed time:timestamp - lag_30']]

X_train = df_train[['concept:name', 
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'next concept:name rfc', 
                    'year', 
                    'month', 
                    'day', 
                    'hour',
                    'CreditScore', 
                    'org:resource',
                    'elapsed time:timestamp - lag_1',
                    'elapsed time:timestamp - lag_2',
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10',
                    'elapsed time:timestamp - lag_11',
                    'elapsed time:timestamp - lag_12',
                    'elapsed time:timestamp - lag_13',
                    'elapsed time:timestamp - lag_14',
                    'elapsed time:timestamp - lag_15',
                    'elapsed time:timestamp - lag_16',
                    'elapsed time:timestamp - lag_17',
                    'elapsed time:timestamp - lag_18',
                    'elapsed time:timestamp - lag_19',
                    'elapsed time:timestamp - lag_20',
                    'elapsed time:timestamp - lag_21',
                    'elapsed time:timestamp - lag_22',
                    'elapsed time:timestamp - lag_23',
                    'elapsed time:timestamp - lag_24',
                    'elapsed time:timestamp - lag_25',
                    'elapsed time:timestamp - lag_26',
                    'elapsed time:timestamp - lag_27',
                    'elapsed time:timestamp - lag_28',
                    'elapsed time:timestamp - lag_29',
                    'elapsed time:timestamp - lag_30']]

X_test = df_test[['concept:name', 
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'next concept:name rfc', 
                    'year', 
                    'month', 
                    'day', 
                    'hour',
                    'CreditScore',
                    'org:resource',
                    'elapsed time:timestamp - lag_1', 
                    'elapsed time:timestamp - lag_2', 
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10',
                    'elapsed time:timestamp - lag_11',
                    'elapsed time:timestamp - lag_12',
                    'elapsed time:timestamp - lag_13',
                    'elapsed time:timestamp - lag_14',
                    'elapsed time:timestamp - lag_15',
                    'elapsed time:timestamp - lag_16',
                    'elapsed time:timestamp - lag_17',
                    'elapsed time:timestamp - lag_18',
                    'elapsed time:timestamp - lag_19',
                    'elapsed time:timestamp - lag_20',
                    'elapsed time:timestamp - lag_21',
                    'elapsed time:timestamp - lag_22',
                    'elapsed time:timestamp - lag_23',
                    'elapsed time:timestamp - lag_24',
                    'elapsed time:timestamp - lag_25',
                    'elapsed time:timestamp - lag_26',
                    'elapsed time:timestamp - lag_27',
                    'elapsed time:timestamp - lag_28',
                    'elapsed time:timestamp - lag_29',
                    'elapsed time:timestamp - lag_30']]

y_train = df_train['elapsed time:timestamp']

y_test = df_test[['elapsed time:timestamp']]

print(f"""
    ###################################### XGBOOST MODEL INFO ######################################\n
      inputs: {[col for col in X_test.columns]} \n
      target: {[col for col in y_test.columns]} \n
    
""")
# Define the parameter grid

param_grid = {
    'n_estimators': [85],
    'max_depth': [7],
    'learning_rate': [0.099]
}
# Initialize the model
model = XGBRegressor(objective='reg:squarederror')

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV using the transformed training set
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print(f"""
      best params (XGBoost): {best_params}\n
""")

# Save the best model
model_filename = 'model_weights/xgboost.joblib'
dump(model, model_filename)
        
# Predict on the test set using the best estimator
y_pred_test = np.abs(best_estimator.predict(X_test))

# Evaluate the model on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_score_value = r2_score(y_test, y_pred_test)

# Update the dataframe with predictions from the best model
df['elapsed time:timestamp XGBoost'] = np.abs(np.where(best_estimator.predict(X) < 0, np.abs(best_estimator.predict(X)) / 10000, best_estimator.predict(X) / 1000))
df['elapsed time:timestamp'] = df['elapsed time:timestamp'].mask(df['elapsed time:timestamp'] < 0)
df = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'year', 'month', 'day', 'hour', 'elapsed time:timestamp - lag_10', 'elapsed time:timestamp - lag_9', 'elapsed time:timestamp - lag_8', 'elapsed time:timestamp - lag_7', 'elapsed time:timestamp - lag_6', 'elapsed time:timestamp - lag_5', 'elapsed time:timestamp - lag_4', 'elapsed time:timestamp - lag_3', 'elapsed time:timestamp - lag_2', 'elapsed time:timestamp - lag_1', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3',
                    'elapsed time:timestamp - lag_11',
                    'elapsed time:timestamp - lag_12',
                    'elapsed time:timestamp - lag_13',
                    'elapsed time:timestamp - lag_14',
                    'elapsed time:timestamp - lag_15',
                    'elapsed time:timestamp - lag_16',
                    'elapsed time:timestamp - lag_17',
                    'elapsed time:timestamp - lag_18',
                    'elapsed time:timestamp - lag_19',
                    'elapsed time:timestamp - lag_20',
                    'elapsed time:timestamp - lag_21',
                    'elapsed time:timestamp - lag_22',
                    'elapsed time:timestamp - lag_23',
                    'elapsed time:timestamp - lag_24',
                    'elapsed time:timestamp - lag_25',
                    'elapsed time:timestamp - lag_26',
                    'elapsed time:timestamp - lag_27',
                    'elapsed time:timestamp - lag_28',
                    'elapsed time:timestamp - lag_29',
                    'elapsed time:timestamp - lag_30'], errors='ignore')
df['elapsed time:timestamp'] = pd.to_timedelta(df['elapsed time:timestamp'], unit='seconds')
df['elapsed time:timestamp XGBoost'] = pd.to_timedelta(df['elapsed time:timestamp XGBoost'], unit='seconds')
df['next time:timestamp XGBoost'] = df['time:timestamp'] + df['elapsed time:timestamp XGBoost']
df = df.drop(columns=['elapsed time:timestamp XGBoost', 'elapsed time:timestamp'], errors='ignore')

# Decode categorical columns
for column in df.columns:
    if column in ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'next concept:name rfc', 'org:resource']:
        le = preprocessors[column]
        df[column] = le.inverse_transform(df[column])
    else:
        continue 

df.to_csv(f'data/{chosed_dataset}_rfc_xgboost.csv')


print(f"""
    ######################################## MODELS RESULTS ########################################\n
      Next Event Prediction Accuracy (Random Forest Classifier):  {rfc_score}\n
      Next Time Prediction R\u00B2 score (XGBoost Regressor): {r2_score_value}\n
      RMSE Time Prediction (XGBoost): {rmse_test}\n
    ################################################################################################\n
""")