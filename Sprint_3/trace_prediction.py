import pickle
import os 
import random
import sys

import pandas as pd
import numpy as np

from joblib import dump
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import path_to_data_folder, slice_index, chosen_dataset, slice_traces
from Sprint_2.train_test_split import train_test_split_custom

df = pd.read_csv(f'data/BPI_Challenge_2017_rfc_xgboost.csv')[:500000]

events_types = df['next concept:name'].unique()

df_traces = pd.read_json(f'data/traces_{chosen_dataset}.json').tail(slice_traces)

# encoding of the events
label_encoder = LabelEncoder()
encoded_event_types = label_encoder.fit_transform(np.append(events_types, ['END', 'absent', 'A_Create Application', 'W_Shortened completion ']))
end_token = label_encoder.transform(['END'])[0]

# MODELS USED FOR THE TRACE PREDICTIONS 

# Prepare data

df_train, df_test = train_test_split_custom(df=df, test_size=0.2, lags=True)

df['concept:name - lag_1'] = df.groupby('case:concept:name')['concept:name'].shift(1).fillna('absent')
df['concept:name - lag_2'] = df.groupby('case:concept:name')['concept:name'].shift(2).fillna('absent')
df['concept:name - lag_3'] = df.groupby('case:concept:name')['concept:name'].shift(3).fillna('absent')
df['concept:name - lag_4'] = df.groupby('case:concept:name')['concept:name'].shift(4).fillna('absent')
df['concept:name - lag_5'] = df.groupby('case:concept:name')['concept:name'].shift(5).fillna('absent')
df['concept:name - lag_6'] = df.groupby('case:concept:name')['concept:name'].shift(6).fillna('absent')
df['concept:name - lag_7'] = df.groupby('case:concept:name')['concept:name'].shift(7).fillna('absent')
df['concept:name - lag_8'] = df.groupby('case:concept:name')['concept:name'].shift(8).fillna('absent')
df['concept:name - lag_9'] = df.groupby('case:concept:name')['concept:name'].shift(9).fillna('absent')
df['concept:name - lag_10'] = df.groupby('case:concept:name')['concept:name'].shift(10).fillna('absent')

columns = ['concept:name' , 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'next concept:name']

for column in columns:
        df_test[column] = label_encoder.transform(df_test[column])
        df_train[column] = label_encoder.transform(df_train[column])
        df[column] = label_encoder.transform(df[column])
        

X_train = df_train[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10']]
X_test = df_test[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10']]

y_train = df_train[['next concept:name']]
y_test = df_test[['next concept:name']]


if not os.path.exists('model_weights/random_forest_trace.pkl'):

    rf_clf = RandomForestClassifier(n_jobs=-1)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [23, 30, 40],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 10, 15],
        'min_samples_leaf': [1, 2, 4]
    }

    inner_cv = KFold(n_splits=5, shuffle=False, random_state=None)  

    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, scoring='accuracy', cv=inner_cv)
    grid_search.fit(X_train, y_train.values.ravel())

    # Get the best model
    rfc_model = grid_search.best_estimator_

    # Print the results

    print(f"""
        Next label prediction in trace:\n
        Score on the test set {rfc_model.score(X_test, y_test)}
        Best parameters: {grid_search.best_params_}
    """)

    # Save the best model to a file using pickle
    # Create the folder for model weights
    os.makedirs('model_weights', exist_ok=True)
    model_filename = 'model_weights/random_forest_trace.pkl'

    with open(model_filename, 'wb') as model_file:
        pickle.dump(rfc_model, model_file)
        
else: 
    with open('model_weights/random_forest_trace.pkl', 'rb') as f:
        rfc_model = pickle.load(f)

for column in columns:
    df[column] = label_encoder.inverse_transform(df[column])

# counting elapsed time
df['elapsed time:timestamp'] = df['time:timestamp diff'].shift(-1) 
df['elapsed time:timestamp'] = pd.to_timedelta(df['elapsed time:timestamp'])
df['elapsed time:timestamp'] = df['elapsed time:timestamp'].apply(lambda x: x.total_seconds()).fillna(-0.01)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

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

# lags for event
df['concept:name - lag_1'] = df.groupby('case:concept:name')['concept:name'].shift(1).fillna('absent')
df['concept:name - lag_2'] = df.groupby('case:concept:name')['concept:name'].shift(2).fillna('absent')
df['concept:name - lag_3'] = df.groupby('case:concept:name')['concept:name'].shift(3).fillna('absent')
df['concept:name - lag_4'] = df.groupby('case:concept:name')['concept:name'].shift(4).fillna('absent')
df['concept:name - lag_5'] = df.groupby('case:concept:name')['concept:name'].shift(5).fillna('absent')
df['concept:name - lag_6'] = df.groupby('case:concept:name')['concept:name'].shift(6).fillna('absent')
df['concept:name - lag_7'] = df.groupby('case:concept:name')['concept:name'].shift(7).fillna('absent')
df['concept:name - lag_8'] = df.groupby('case:concept:name')['concept:name'].shift(8).fillna('absent')
df['concept:name - lag_9'] = df.groupby('case:concept:name')['concept:name'].shift(9).fillna('absent')
df['concept:name - lag_10'] = df.groupby('case:concept:name')['concept:name'].shift(10).fillna('absent')

# preprocess the columns before fitting
for column in df.columns:
    if column == 'time:timestamp':
        df['year'] = df['time:timestamp'].dt.year
        df['month'] = df['time:timestamp'].dt.month
        df['day'] = df['time:timestamp'].dt.day
        df['hour'] = df['time:timestamp'].dt.hour

    elif column in ['next concept:name rfc', 'concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10']:
        df[column] = label_encoder.transform(df[column])

    else:
        continue
    
# split the data on train and test dataframes 
df_train, df_test = train_test_split_custom(df=df, lags=False, test_size=0.2)

# define the input and outputs for the model before put in the train 

X = df[['next concept:name rfc', 'concept:name',  # this X is for making prediction on the dataframe for saving results in the csv
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'concept:name - lag_4', 
                    'concept:name - lag_5', 
                    'concept:name - lag_6', 
                    'concept:name - lag_7', 
                    'concept:name - lag_8', 
                    'concept:name - lag_9', 
                    'concept:name - lag_10', 
                    'elapsed time:timestamp - lag_1', 
                    'elapsed time:timestamp - lag_2', 
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10']]

X_train = df_train[['next concept:name rfc', 'concept:name', 
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'concept:name - lag_4', 
                    'concept:name - lag_5', 
                    'concept:name - lag_6', 
                    'concept:name - lag_7', 
                    'concept:name - lag_8', 
                    'concept:name - lag_9', 
                    'concept:name - lag_10',  
                    'elapsed time:timestamp - lag_1',
                    'elapsed time:timestamp - lag_2',
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10']]

X_test = df_test[['next concept:name rfc', 'concept:name', 
                    'concept:name - lag_1', 
                    'concept:name - lag_2', 
                    'concept:name - lag_3', 
                    'concept:name - lag_4', 
                    'concept:name - lag_5', 
                    'concept:name - lag_6', 
                    'concept:name - lag_7', 
                    'concept:name - lag_8', 
                    'concept:name - lag_9', 
                    'concept:name - lag_10',  
                    'elapsed time:timestamp - lag_1', 
                    'elapsed time:timestamp - lag_2', 
                    'elapsed time:timestamp - lag_3',
                    'elapsed time:timestamp - lag_4',
                    'elapsed time:timestamp - lag_5',
                    'elapsed time:timestamp - lag_6',
                    'elapsed time:timestamp - lag_7',
                    'elapsed time:timestamp - lag_8',
                    'elapsed time:timestamp - lag_9',
                    'elapsed time:timestamp - lag_10']]

y_train = df_train['elapsed time:timestamp']

y_test = df_test[['elapsed time:timestamp']]

# Define the parameter grid

param_grid = {
    'n_estimators': [85],
    'max_depth': [7],
    'learning_rate': [0.099]
}
# Initialize the model
model = XGBRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV using the transformed training set
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
xgboost_trace = grid_search.best_estimator_

print(f"best params: {best_params}")

# Save the best model
model_filename = 'model_weights/xgboost_trace.joblib'
dump(xgboost_trace, model_filename)
        
# Predict on the test set using the best estimator
y_pred_test = np.abs(xgboost_trace.predict(X_test))

# Evaluate the model on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'Test RMSE: {rmse_test}')

r2_score_value = r2_score(y_test, y_pred_test)
print(f'R² score: {r2_score_value}')

# Update the dataframe with predictions from the best model
df['elapsed time:timestamp XGBoost'] = np.abs(np.where(xgboost_trace.predict(X) < 0, np.abs(xgboost_trace.predict(X)) / 10000, xgboost_trace.predict(X) / 1000))
df['elapsed time:timestamp'] = df['elapsed time:timestamp'].mask(df['elapsed time:timestamp'] < 0)
df = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'year', 'month', 'day', 'hour', 'elapsed time:timestamp - lag_10', 'elapsed time:timestamp - lag_9', 'elapsed time:timestamp - lag_8', 'elapsed time:timestamp - lag_7', 'elapsed time:timestamp - lag_6', 'elapsed time:timestamp - lag_5', 'elapsed time:timestamp - lag_4', 'elapsed time:timestamp - lag_3', 'elapsed time:timestamp - lag_2', 'elapsed time:timestamp - lag_1', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10'], errors='ignore')
df['elapsed time:timestamp'] = pd.to_timedelta(df['elapsed time:timestamp'], unit='seconds')
df['elapsed time:timestamp XGBoost'] = pd.to_timedelta(df['elapsed time:timestamp XGBoost'], unit='seconds')
df['next time:timestamp XGBoost'] = df['time:timestamp'] + df['elapsed time:timestamp XGBoost']
df = df.drop(columns=['elapsed time:timestamp XGBoost', 'elapsed time:timestamp'], errors='ignore')

# Decode categorical columns
for column in df.columns:
    if column in ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10']:
        df[column] = label_encoder.inverse_transform(df[column])
    else:
        continue 

predicted_traces_list = []
case_concept_names_list = []

with open('model_weights/random_forest_trace.pkl', 'rb') as f:
        rfc_model = pickle.load(f)


# RUN THE RECURRENT STRUCTURE ON THE TRACES
for index, row in df_traces.iterrows():

      case_concept_name = row['case:concept:name']
      df_trace = pd.DataFrame(row['trace'])

      df_trace['concept:name - lag_1'] = df_trace['concept:name'].shift(1).fillna('absent')
      df_trace['concept:name - lag_2'] = df_trace['concept:name'].shift(2).fillna('absent')
      df_trace['concept:name - lag_3'] = df_trace['concept:name'].shift(3).fillna('absent')
      df_trace['concept:name - lag_4'] = df_trace['concept:name'].shift(4).fillna('absent')
      df_trace['concept:name - lag_5'] = df_trace['concept:name'].shift(5).fillna('absent')
      df_trace['concept:name - lag_6'] = df_trace['concept:name'].shift(6).fillna('absent')
      df_trace['concept:name - lag_7'] = df_trace['concept:name'].shift(7).fillna('absent')
      df_trace['concept:name - lag_8'] = df_trace['concept:name'].shift(8).fillna('absent')
      df_trace['concept:name - lag_9'] = df_trace['concept:name'].shift(9).fillna('absent')
      df_trace['concept:name - lag_10'] = df_trace['concept:name'].shift(10).fillna('absent')

      # find out the number of rows in the trace
      num_rows = len(df_trace)

      # Choose the upper bound for the random integer
      upper_bound = min(num_rows, 10)

      # Generate a random integer within the specified range
      random_index = random.randint(1, upper_bound)

      # slice the trace on random integer
      df_trace = df_trace.iloc[:random_index]
      
      # encode categorical columns in the trace 
      columns = ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10']
      for column in columns:
            df_trace[column] = label_encoder.transform(df_trace[column])
            
      
      event_rfc = df_trace[columns].iloc[[-1]].values

      # build the suffix dataframe for concept:name
      predicted_suffix = pd.DataFrame({'concept:name': [], 'time:timestamp': []})
      operator = True
      while operator:
            window = pd.DataFrame([event_rfc[:, :11][0].tolist()], columns=['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10'])
            prediction_next_event = rfc_model.predict(window)
            event_rfc = np.insert(event_rfc, 0, prediction_next_event, axis=1)

            if (event_rfc[0,0] == end_token) and event_rfc.shape[1] > 15:
                  operator = False

            if event_rfc.shape[1] >= 80:
                  operator = False
                  
      # build the suffix dataframe for concept:name
      predicted_suffix['concept:name'] = event_rfc.tolist()[0][::-1]
      predicted_suffix['time:timestamp'] = [np.nan] * len(event_rfc.tolist()[0][::-1])

      # join the suffix dataframe with the trace
      df_trace = pd.concat([df_trace, predicted_suffix], ignore_index=True, axis=0)

      # define the next event
      df_trace['next concept:name rfc'] = df_trace['concept:name'].shift(-1)

      # encode categorical columns in the next event
      df_trace.loc[len(df_trace)-1, 'next concept:name rfc'] = label_encoder.transform(['END'])
      encoding_for_absent = label_encoder.transform(['absent'])[0]

      # computing lags for concept:name again 
      df_trace['concept:name - lag_2'] = df_trace['concept:name'].shift(2).fillna(encoding_for_absent)
      df_trace['concept:name - lag_3'] = df_trace['concept:name'].shift(3).fillna(encoding_for_absent)
      df_trace['concept:name - lag_1'] = df_trace['concept:name'].shift(1).fillna(encoding_for_absent)
      df_trace['concept:name - lag_4'] = df_trace['concept:name'].shift(4).fillna(encoding_for_absent)
      df_trace['concept:name - lag_5'] = df_trace['concept:name'].shift(5).fillna(encoding_for_absent)
      df_trace['concept:name - lag_6'] = df_trace['concept:name'].shift(6).fillna(encoding_for_absent)
      df_trace['concept:name - lag_7'] = df_trace['concept:name'].shift(7).fillna(encoding_for_absent)
      df_trace['concept:name - lag_8'] = df_trace['concept:name'].shift(8).fillna(encoding_for_absent)
      df_trace['concept:name - lag_9'] = df_trace['concept:name'].shift(9).fillna(encoding_for_absent)
      df_trace['concept:name - lag_10'] = df_trace['concept:name'].shift(10).fillna(encoding_for_absent)

      # counting elapsed time
      df_trace['time:timestamp'] = pd.to_datetime(df_trace['time:timestamp'], format='mixed')
      df_trace['time:timestamp diff'] = df_trace['time:timestamp'].diff()
      df_trace['elapsed time:timestamp'] = df_trace['time:timestamp diff'].shift(-1) 
      df_trace['elapsed time:timestamp'] = pd.to_timedelta(df_trace['elapsed time:timestamp'])
      df_trace['elapsed time:timestamp'] = df_trace['elapsed time:timestamp'].apply(lambda x: x.total_seconds()).fillna(-0.01)

      # counting lags for elapsed time again 
      df_trace['elapsed time:timestamp - lag_1'] = df_trace['elapsed time:timestamp'].shift(1).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_2'] = df_trace['elapsed time:timestamp'].shift(2).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_3'] = df_trace['elapsed time:timestamp'].shift(3).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_4'] = df_trace['elapsed time:timestamp'].shift(4).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_5'] = df_trace['elapsed time:timestamp'].shift(5).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_6'] = df_trace['elapsed time:timestamp'].shift(6).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_7'] = df_trace['elapsed time:timestamp'].shift(7).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_8'] = df_trace['elapsed time:timestamp'].shift(8).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_9'] = df_trace['elapsed time:timestamp'].shift(9).fillna(-0.00000001)
      df_trace['elapsed time:timestamp - lag_10'] = df_trace['elapsed time:timestamp'].shift(10).fillna(-0.00000001)

      # calculating the time predictions for the trace and suffix
      index = random_index
      xgboost_inputs = df_trace.drop(columns=['time:timestamp diff', 'time:timestamp', 'elapsed time:timestamp'], axis=1).loc[index]
      while index < len(df_trace) -1:
            xgboost_inputs = xgboost_inputs[['next concept:name rfc', 'concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'elapsed time:timestamp - lag_1', 'elapsed time:timestamp - lag_2', 'elapsed time:timestamp - lag_3', 'elapsed time:timestamp - lag_4', 'elapsed time:timestamp - lag_5', 'elapsed time:timestamp - lag_6', 'elapsed time:timestamp - lag_7', 'elapsed time:timestamp - lag_8', 'elapsed time:timestamp - lag_9', 'elapsed time:timestamp - lag_10']]
            df_trace.loc[index, 'time:timestamp'] = df_trace.loc[index-1, 'time:timestamp'] + pd.to_timedelta(np.abs(np.where(xgboost_trace.predict(pd.DataFrame(xgboost_inputs).transpose())[0] < 0, np.abs(xgboost_trace.predict(pd.DataFrame(xgboost_inputs).transpose())[0]) / 10000, xgboost_trace.predict(pd.DataFrame(xgboost_inputs).transpose())[0] / 1000)), unit='s')
            index = index + 1
      
      # take only needed columns
      df_trace = df_trace[['concept:name', 'time:timestamp']]

      # convert time to string format
      df_trace['time:timestamp'] = df_trace['time:timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

      # cut off the end token 
      df_trace = df_trace.drop(df_trace.index[-1])
      df_trace['concept:name'] = label_encoder.inverse_transform(df_trace['concept:name'])

      # append the trace to the list with corresponding case:concept:name
      predicted_traces_list.append(df_trace)
      case_concept_names_list.append(case_concept_name)     
      
      # save the final predictions
      df_final_predictions = pd.DataFrame({'case:concept:name': case_concept_names_list, 'predicted_traces': predicted_traces_list})

# save the final predictions in the json file
df_final_predictions.to_json(f'data/trace_predictions_{chosen_dataset}.json', orient='records')

