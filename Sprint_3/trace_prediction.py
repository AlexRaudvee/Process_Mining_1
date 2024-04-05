import pickle
import os 
import random
import sys
import json
import editdistance

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from joblib import dump
from ast import literal_eval
from collections import Counter
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import path_to_data_folder, slice_index, chosen_dataset, slice_traces
from Sprint_2.train_test_split import train_test_split_custom
from func import jaccard_similarity, extract_events, safely_parse_json, print_terminal_width_symbol, print_centered_text

pd.set_option('future.no_silent_downcasting', True)

print_terminal_width_symbol('#')
print('\n')
print_centered_text('TRACE PREDICTION RUNNING')
print('\n')

# artifacts folder creation
os.makedirs(f'{os.getcwd()}/artifacts_traces_model', exist_ok=True)
path_to_artifacts = f'{os.getcwd()}/artifacts_traces_model'

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

# Save the best model
model_filename = 'model_weights/xgboost_trace.joblib'
dump(xgboost_trace, model_filename)
        
# Predict on the test set using the best estimator
y_pred_test = np.abs(xgboost_trace.predict(X_test))

mae_test = mean_absolute_error(y_test, y_pred_test)

# Evaluate the model on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))


r2_score_value = r2_score(y_test, y_pred_test)


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

# load files
with open('../data/trace_predictions_BPI_Challenge_2017.json', 'r') as f:
    predicted_traces = json.load(f)

with open('../data/traces_BPI_Challenge_2017.json', 'r') as f:
    actual_traces = json.load(f)

# for matching 1000 'case:concept:name'
matched_traces = []

for pred_trace in predicted_traces:
    case_name = pred_trace['case:concept:name']
    actual_trace = next(
        (trace for trace in actual_traces if trace['case:concept:name'] == case_name), None)
    if actual_trace:
        matched_traces.append({
            'case:concept:name': case_name,
            'predicted_traces': pred_trace['predicted_traces'],
            'trace': actual_trace['trace']
        })

results_df = pd.DataFrame(matched_traces)
results_df.to_csv(
    '../data/matched_traces.csv', index=False)

df = pd.read_csv(
    '../data/matched_traces.csv')

print_terminal_width_symbol(symbol='#')
print('\n')
print_centered_text("METRICS FOR TRACE PREDICTION MODEL (RFC & XGBoost)")
print('\n')
print(f"""
        R\u00B2 of Time Prediction (Trace): {r2_score_value}\n  
        MAE Time Prediction (Trace): {mae_test/60/60} in hours \n
""")
print_terminal_width_symbol(symbol='#')

# Initialize lists to hold the metrics for each trace pair, Initialize a dictionary to hold the Edit Distances and case identifiers
edit_distances = []
jaccard_similarities = []
case_edit_distances = {}

# Loop through each row and calculate metrics
for _, row in df.iterrows():
    predicted_events = extract_events(row['predicted_traces'])
    actual_events = extract_events(row['trace'])

    # Calculate the edit distance for this pair of traces
    distance = editdistance.eval(predicted_events, actual_events)
    edit_distances.append(distance)
    # Add the distance with the case identifier to the dictionary
    case_edit_distances[row['case:concept:name']] = distance
    # Calculate the Jaccard similarity for this pair of traces
    similarity = jaccard_similarity(set(predicted_events), set(actual_events))
    jaccard_similarities.append(similarity)

# Calculate the mean of the edit distances and Jaccard similarities
mean_edit_distance = np.mean(edit_distances)
mean_jaccard_similarity = np.mean(jaccard_similarities)

# Find the case with the minimum Edit Distance
min_distance_case = min(case_edit_distances, key=case_edit_distances.get)
min_distance = case_edit_distances[min_distance_case]

print_terminal_width_symbol(symbol='#')
print('\n')
print_centered_text("ADVANCED METRICS FOR TRACE PREDICTION + VISUALIZATIONS")
print('\n')
print(f"""
      Minimum Edit Distance: {min_distance} for case: {min_distance_case}\n
      Mean Edit Distance for the entire dataset: {mean_edit_distance}\n
      Mean Jaccard Similarity for the entire dataset: {mean_jaccard_similarity}
""")


# Assuming the DataFrame is loaded from a CSV file:
df = pd.read_csv('../data/matched_traces.csv',
                 converters={'predicted_traces': literal_eval, 'trace': literal_eval})

# Define a function to extract events from the serialized lists
def extract_events(serialized_list):
    return [event['concept:name'] for event in serialized_list]


# Apply this function to your columns to create the 'predicted_events' and 'actual_events' columns
df['predicted_events'] = df['predicted_traces'].apply(extract_events)
df['actual_events'] = df['trace'].apply(extract_events)

# Flatten the lists to count the frequency of each event
predicted_flattened = [event for sublist in df['predicted_events'] for event in sublist]
actual_flattened = [event for sublist in df['actual_events'] for event in sublist]

predicted_counts = Counter(predicted_flattened)
actual_counts = Counter(actual_flattened)

# Convert counters to dataframes
predicted_df = pd.DataFrame(predicted_counts.items(), columns=[
                            'Event', 'Predicted Frequency'])
actual_df = pd.DataFrame(actual_counts.items(), columns=[
                         'Event', 'Actual Frequency'])

# Merge on Event
merged_df = pd.merge(predicted_df, actual_df, on='Event')

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
width = 0.35  # bar width
ind = np.arange(len(merged_df))  # the x locations for the groups

ax.barh(ind - width/2,
        merged_df['Predicted Frequency'], width, label='Predicted')
ax.barh(ind + width/2, merged_df['Actual Frequency'], width, label='Actual')

ax.set(yticks=ind, yticklabels=merged_df['Event'], ylim=[
       2*width - 1, len(merged_df)])
ax.legend()
plt.title("compares the frequency of each event between the predicted traces and actual traces", fontsize=14)
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

# Over forecast (predicted bars are longer than they actually are) and under forecast (predicted bars are shorter than they actually are)

def create_sankey_df(events_list):
    source, target, value = [], [], []
    for trace in events_list:
        for i in range(len(trace)-1):
            if (trace[i], trace[i+1]) not in list(zip(source, target)):
                source.append(trace[i])
                target.append(trace[i+1])
                value.append(1)
            else:
                idx = list(zip(source, target)).index((trace[i], trace[i+1]))
                value[idx] += 1
    return pd.DataFrame({'source': source, 'target': target, 'value': value})


# Create DataFrames for predicted and actual events
predicted_sankey = create_sankey_df(df['predicted_events'])
actual_sankey = create_sankey_df(df['actual_events'])

# Generate a list of all unique events
all_events = list(set(predicted_sankey['source'].tolist() + predicted_sankey['target'].tolist() + actual_sankey['source'].tolist() + actual_sankey['target'].tolist()))
event_dict = {event: i for i, event in enumerate(all_events)}

# Map events to integers
predicted_sankey.replace(event_dict, inplace=True)
actual_sankey.replace(event_dict, inplace=True)

# Plot
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_events,
    ),
    link=dict(
        source=predicted_sankey['source'],
        target=predicted_sankey['target'],
        value=predicted_sankey['value'],
        color="blue"
    ))])

fig.update_layout(title_text="Predicted Trace Flows", font_size=10)
fig.write_image(os.path.join(path_to_artifacts, "Predicted_Trace_Flows.png"))

# Apply the function to parse JSON for both columns if they're not already parsed
df['predicted_traces'] = df['predicted_traces'].apply(safely_parse_json)
df['trace'] = df['trace'].apply(safely_parse_json)

# Randomly select a trace
selected_index = random.randint(0, len(df) - 1)
selected_trace = df.iloc[selected_index]

# Extract the case name
case_name = selected_trace['case:concept:name']

# Extract and prepare data for plotting
predicted_events = [event['concept:name'] for event in selected_trace['predicted_traces']]
actual_events = [event['concept:name'] for event in selected_trace['trace']]
# Create mappings for events to integers
all_events = set(predicted_events + actual_events)
event_to_int = {event: i for i, event in enumerate(all_events)}

# Convert events to integers for plotting
mapped_pred_seq = [event_to_int[event] for event in predicted_events]
mapped_act_seq = [event_to_int[event] for event in actual_events]

# Generate the plot
fig, ax = plt.subplots(figsize=(25, 10))

# Plotting both predicted and actual sequences
ax.plot(mapped_pred_seq, 'o-', label='Predicted', color='blue')
ax.plot(mapped_act_seq, 'x-', label='Actual', color='red')

# Set the ticks and labels
ax.set_yticks(list(event_to_int.values()))
ax.set_yticklabels(list(event_to_int.keys()))
ax.legend()

plt.xlabel('Sequence Position', fontsize=14)
plt.xticks(np.arange(max(len(predicted_events), len(actual_events))))
plt.title(f' Comparison of Predicted and Actual Sequences for {case_name}', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

# Find the corresponding row for the case with the minimum Edit Distance
min_distance_row = df[df['case:concept:name'] == min_distance_case].iloc[0]

# Extract events for the minimum Edit Distance case
min_predicted_events = extract_events(min_distance_row['predicted_traces'])
min_actual_events = extract_events(min_distance_row['trace'])

# Now you can generate a comparison plot for the selected case
fig, ax = plt.subplots(figsize=(25, 10))

# Defining a simple mapping from event names to integers
event_set = set(min_predicted_events + min_actual_events)
event_map = {event: i for i, event in enumerate(event_set)}

# Mapping events to integers for plotting
mapped_min_pred_seq = [event_map[event] for event in min_predicted_events]
mapped_min_act_seq = [event_map[event] for event in min_actual_events]

# Plotting both predicted and actual sequences for the case with minimum Edit Distance
ax.plot(mapped_min_pred_seq, marker='o', label='Predicted', linestyle='-', color='blue')
ax.plot(mapped_min_act_seq, marker='x', label='Actual', linestyle='-', color='red')

ax.set_yticks(np.arange(len(event_set)))
ax.set_yticklabels(event_map.keys())
ax.legend()

plt.xlabel('Sequence Position', fontsize=14)
plt.title(f' Comparison of Predicted and Actual Sequences for {min_distance_case}', fontsize=16)
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

# Flatten the list of events for all sequences
all_pred_events = [event for sublist in df['predicted_events'] for event in sublist]
all_act_events = [event for sublist in df['actual_events'] for event in sublist]

pred_counts = Counter(all_pred_events)
act_counts = Counter(all_act_events)

# Creating a DataFrame from the counters
event_df = pd.DataFrame({
    'Predicted': pred_counts,
    'Actual': act_counts
})

# Filling NaN values with zeros since they represent no occurrence
event_df = event_df.fillna(0)

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(event_df, annot=True, fmt='.0f', cmap='viridis')
plt.title('Heatmap of Event Occurrence', fontsize=16)
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))
# Assess whether events are overestimated, underestimated, or closely match actual counts

pred_lengths = df['predicted_events'].apply(len)
act_lengths = df['actual_events'].apply(len)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(pred_lengths, bins=20, alpha=0.5, label='Predicted Lengths')
plt.hist(act_lengths, bins=20, alpha=0.5, label='Actual Lengths')
plt.xlabel('Sequence Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Sequence Lengths', fontsize=16)
plt.legend()

# If there are bars that extend far to the right, these bars may represent unusually long sequences of outliers
# significant differences between the distribution of predicted and actual sequence lengths. The predicted sequence appears to cluster around a specific length, while the actual sequence shows a different pattern, indicating that the actual sequence length varies more

pred_lengths = df['predicted_events'].apply(len)
act_lengths = df['actual_events'].apply(len)

# Plot the cumulative distribution
plt.figure(figsize=(10, 6))
plt.hist(pred_lengths, bins=range(min(pred_lengths), max(pred_lengths) + 1), alpha=0.5, label='Predicted Lengths', cumulative=True, histtype='step', linewidth=2)
plt.hist(act_lengths, bins=range(min(act_lengths), max(act_lengths) + 1), alpha=0.5, label='Actual Lengths', cumulative=True, histtype='step', linewidth=2)
plt.xlabel('Trace Length', fontsize=15)
plt.ylabel('Cumulative Frequency', fontsize=15)
plt.title('Cumulative Distribution of Trace Lengths', fontsize=20)
plt.legend(fontsize=12)
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

# Extract events and timestamps from the 'predicted_traces' and 'trace' columns
df['predicted_events'] = df['predicted_traces'].apply(lambda x: [event['concept:name'] for event in x])
df['actual_events'] = df['trace'].apply(lambda x: [event['concept:name'] for event in x])
df['predicted_timestamps'] = df['predicted_traces'].apply(lambda x: [event['time:timestamp'] for event in x])
df['actual_timestamps'] = df['trace'].apply(lambda x: [event['time:timestamp'] for event in x])

# Count the frequency of each predicted event
all_predicted_events = [event for sublist in df['predicted_events'].tolist() for event in sublist]
event_counter = Counter(all_predicted_events)

# Create a DataFrame from the counter
event_freq_df = pd.DataFrame(list(event_counter.items()), columns=['Event', 'Frequency'])

event_freq_df = event_freq_df[event_freq_df['Event'] != 'END']

# Plot the frequency of events as a bar chart
plt.figure(figsize=(20, 10))
sns.barplot(x='Frequency', y='Event', data=event_freq_df.sort_values('Frequency', ascending=False))
plt.title('Frequency of Predicted Events', fontsize=20)
plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Event', fontsize=15)
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

print_terminal_width_symbol('#')