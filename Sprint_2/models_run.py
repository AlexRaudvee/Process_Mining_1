import pickle
import os 
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from tqdm import tqdm
from joblib import dump
from train_test_split import train_test_split_custom
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import path_to_data_folder, slice_index, chosen_dataset
from Sprint_1.cleaning_feature_extraction import print_centered_text, print_terminal_width_symbol

# artifacts folder creation
os.makedirs(f'{os.getcwd()}/artifacts_models', exist_ok=True)
path_to_artifacts = f'{os.getcwd()}/artifacts_models'

# functions for outputs
custom_format = "{desc}: {percentage:.0f}%\x1b[33m|\x1b[0m\x1b[32m{bar}\x1b[0m\x1b[31m{remaining}\x1b[0m\x1b[33m|\x1b[0m {n}/{total} [{elapsed}<{remaining}]"

def saver(df: pd.DataFrame, path_name: str):

    chunks = np.array_split(df.index, 100) # split into 100 chunks

    for chunck, subset in enumerate(tqdm(chunks, desc=f"Storing of data ", dynamic_ncols=True, bar_format=custom_format, ascii=' -')):
        if chunck == 0: # first row
            df.loc[subset].to_csv(path_name, mode='w', index=True)
        else:
            df.loc[subset].to_csv(path_name, header=None, mode='a', index=True)

### NAIVE MODEL CODE
df = pd.read_csv(f'data/clean_{chosen_dataset}.csv')[:slice_index]

activity_counts = df['concept:name'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=activity_counts.values, y=activity_counts.index)
plt.xlabel('Frequency')
plt.ylabel('Activity')
plt.title('Activity Frequency')
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))


# Assuming 'time:timestamp' is the timestamp and 'case:concept:name' is the case ID
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
case_durations = df.groupby('case:concept:name')['time:timestamp'].agg([min, max])
case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / 3600  # Duration in hours

plt.figure(figsize=(10, 6))
sns.histplot(case_durations['duration'], bins=30, kde=True)
plt.xlabel('Duration (Hours)')
plt.ylabel('Count')
plt.title('Case Duration Distribution')
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

activities_per_case = df.groupby('case:concept:name').size()

plt.figure(figsize=(10, 6))
sns.histplot(activities_per_case, bins=30, kde=True)
plt.xlabel('Number of Activities')
plt.ylabel('Count')
plt.title('Number of Activities per Case')
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

# Extract prefixes ('W_', 'O_', 'A_')
df['prefix'] = df['concept:name'].str.split('_').str[0]

# Create separate columns for each step
df['W'] = df['prefix'].apply(lambda x: x == 'W')
df['O'] = df['prefix'].apply(lambda x: x == 'O')
df['A'] = df['prefix'].apply(lambda x: x == 'A')

# Plot the frequency of tasks chosen at each step
fig, ax = plt.subplots(figsize=(10, 6))

df.groupby('prefix').size().plot(kind='bar', color=['blue', 'green', 'orange'], ax=ax)
ax.set_title('Frequency of Tasks in Each Step')
ax.set_xlabel('Step')
ax.set_ylabel('Frequency')

# Create a directed graph
G = nx.DiGraph()

# Add edges for transitions between activities for each case
# Here, we'll track the last activity to create a sequence within each case
last_activity = {}

for index, row in df.iterrows():
    case_id = row['case:concept:name']
    activity = row['concept:name']
    
    # Check if the current case had a previous activity
    if case_id in last_activity:
        # Add edge from last activity to current activity for the same case
        G.add_edge(last_activity[case_id], activity)
    
    # Update the last activity for the current case
    last_activity[case_id] = activity

# Use a spring layout to visualize the graph, attempting to reflect some hierarchy or sequence
pos = nx.spring_layout(G, seed=42)  # Seed for reproducible layout

# Drawing the graph
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color="lightblue")
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")

plt.title("Network Graph of Process Activities")
plt.axis("off")  # Turn off the axis
plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

# Generate event sequence numbers within each case
df['event_seq'] = df.groupby('case:concept:name').cumcount() + 1

# Calculate the difference between the current event's timestamp and the next one
df['time_to_next_event'] = df.groupby('case:concept:name')['time:timestamp'].transform(lambda x: x.diff().shift(-1))

# Convert the 'time_to_next_event' from timedelta to seconds (or any other numeric representation you prefer)
df['time_to_next_event_seconds'] = df['time_to_next_event'].dt.total_seconds().fillna(0)

# Step 2: Compute the mean duration for each 'concept:name'
average_durations_per_concept = df[['time_to_next_event_seconds', 'concept:name']].groupby('concept:name').median().rename(columns={'time_to_next_event_seconds': 'mean_duration_seconds'})

# Step 3: Merge this mean duration back into the original dataframe to use as a prediction
df = pd.merge(df, average_durations_per_concept, how='left', on='concept:name')

# Rename the 'mean_duration_seconds' column to something like 'predicted_duration_seconds'
df.rename(columns={'mean_duration_seconds': 'predicted_time_to_next_event_seconds'}, inplace=True)

df[['predicted_time_to_next_event_seconds', 'time_to_next_event_seconds']] = df[['predicted_time_to_next_event_seconds', 'time_to_next_event_seconds']].round(2)

# Optionally, convert the 'predicted_time_to_next_event_seconds' back to a timedelta for readability or further datetime operations
# dataframe_2012['predicted_time_to_next_event'] = pd.to_timedelta(dataframe_2012['predicted_time_to_next_event_seconds'], unit='s')

df = df.drop(columns=['column_similarity_percentage'], errors='ignore')

# Subtract the timestamp of the first event in each case from all events in that case
df['elapsed_time_from_start'] = df.groupby('case:concept:name')['time:timestamp'].transform(lambda x: x - x.min())

# Calculate the average elapsed time from the start for each 'event_seq'
predicted_start_time = df.groupby('event_seq')['elapsed_time_from_start'].mean().reset_index(name='predicted_start_time')

# Merge this average elapsed time back into the original DataFrame to use as a predicted start time
df = pd.merge(df, predicted_start_time, on='event_seq', how='left')

# Calculate the most common concept:name for each event_seq
most_common_concepts_by_seq = df.groupby('event_seq')['concept:name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='predicted_step')

# Merge the most common concept:name for each event_seq back into the original DataFrame
df = pd.merge(df, most_common_concepts_by_seq, on='event_seq', how='left')

df['column_similarity_percentage'] = (df['concept:name'] == df['predicted_step']).mean() * 100

# accuracy for label prediction
accuracy = df['column_similarity_percentage'].loc[0]

# r2 for time prediction
r2_time = r2_score(df['time_to_next_event_seconds'], df['predicted_time_to_next_event_seconds'])

# compute the mean absolute error for time prediction
mae_time = mean_absolute_error(df['time_to_next_event_seconds'], df['predicted_time_to_next_event_seconds'])

# rename columns with predictions
df = df.rename(columns={'predicted_step': 'next concept:name naive', 'predicted_time_to_next_event_seconds': 'next time:timestamp naive', 'time_to_next_event': 'next time:timestamp', 'timestamp_difference': 'time:timestamp diff'}, errors='ignore')

# convert the times back to timestamps
df['next time:timestamp'] = df['time:timestamp'] + df['next time:timestamp']
df['next time:timestamp naive'] = pd.to_timedelta(df['next time:timestamp naive'], unit='s')
df['next time:timestamp naive'] = df['time:timestamp'] + df['next time:timestamp naive']
df['next concept:name naive'] = df.groupby('case:concept:name')['next concept:name naive'].shift(-1)

# drop columns that are not going to be used
df = df.drop(columns=['column_similarity_percentage', 'predicted_start_time', 'elapsed_time_from_start', 'time_to_next_event_seconds', 'event_seq', 'A', 'O', 'W', 'prefix'], errors='ignore')

# save to the csv 
saver(df, f'data/{chosen_dataset}_naive.csv')

print_terminal_width_symbol(symbol='#')
print('\n')
print_centered_text("METRICS FOR BASELINE MODEL (NAIVE)")
print('\n')
print(f"""
        Accuracy Next Event Prediction (Naive): {accuracy}% \n
        R\u00B2 of Time Prediction (Naive): {r2_time}\n  
        MAE Time Prediction (Naive): {mae_time} \n
""")
print_terminal_width_symbol(symbol='#')


# extract the 1-10 lags
df = pd.read_csv(f'data/{chosen_dataset}_naive.csv')

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

# define target
df['next concept:name'] = df.groupby('case:concept:name')['concept:name'].shift(-1).fillna('END')



# Prepare data

df_train, df_test = train_test_split_custom(df=df, test_size=0.2, lags=True)

columns = ['concept:name' , 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'Weekday', 'working_hours', 'vacation_day', 'next concept:name']
label_encoders = {}
for column in columns:
        label_encoder = LabelEncoder()
        df_test[column] = label_encoder.fit_transform(df_test[column])
        df_train[column] = label_encoder.fit_transform(df_train[column])
        df[column] = label_encoder.fit_transform(df[column])
        label_encoders[column] = label_encoder

X_train = df_train[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'Weekday', 'working_hours', 'vacation_day']]
X_test = df_test[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'Weekday', 'working_hours', 'vacation_day']]

y_train = df_train[['next concept:name']]
y_test = df_test[['next concept:name']]

print(f"""
    inputs: {[col for col in X_test.columns]} \n
    target: {[col for col in y_test.columns]}
""")

if not os.path.exists('model_weights/random_forest.pkl'):

    rf_clf = RandomForestClassifier(n_jobs=-1)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [23, 30, 40],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 10, 15],
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
    print(f"""
        Best parameters: {grid_search.best_params_}
    """)

    # Save the best model to a file using pickle
    # Create the folder for model weights
    os.makedirs('model_weights', exist_ok=True)
    model_filename = 'model_weights/random_forest.pkl'

    with open(model_filename, 'wb') as model_file:
        pickle.dump(best_model, model_file)
        
else: 
    rfc_score = 82.622516556291391
    with open('model_weights/random_forest.pkl', 'rb') as f:
        best_model = pickle.load(f)

# Make predictions on the dataset for adding new column
df['next concept:name rfc'] = label_encoder.inverse_transform(best_model.predict(df[['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'Weekday', 'working_hours', 'vacation_day']]))

for column in columns:
        df[column] = label_encoders[column].inverse_transform(df[column])

saver(df, f'data/{chosen_dataset}_rfc_xgboost.csv')

df = pd.read_csv(f'data/{chosen_dataset}_rfc_xgboost.csv')

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


# preprocess the columns before fitting
preprocessors = {}
for column in df.columns:
    if column == 'time:timestamp':
        df['year'] = df['time:timestamp'].dt.year
        df['month'] = df['time:timestamp'].dt.month
        df['day'] = df['time:timestamp'].dt.day
        df['hour'] = df['time:timestamp'].dt.hour

    elif column in ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'next concept:name rfc', 'org:resource', 'Weekday', 'working_hours', 'vacation_day']:
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
                    'concept:name - lag_4', 
                    'concept:name - lag_5', 
                    'concept:name - lag_6', 
                    'concept:name - lag_7', 
                    'concept:name - lag_8', 
                    'concept:name - lag_9', 
                    'concept:name - lag_10', 
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
                    'Weekday', 
                    'working_hours', 
                    'vacation_day']]

X_train = df_train[['concept:name', 
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
                    'Weekday', 
                    'working_hours', 
                    'vacation_day']]

X_test = df_test[['concept:name', 
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
                    'Weekday', 
                    'working_hours', 
                    'vacation_day']]

y_train = df_train['elapsed time:timestamp']

y_test = df_test[['elapsed time:timestamp']]

print(f"""
    inputs: {[col for col in X_test.columns]} \n
    target: {[col for col in y_test.columns]}
""")
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
best_estimator = grid_search.best_estimator_

print(f"best params: {best_params}")

# Save the best model
model_filename = 'model_weights/xgboost.joblib'
dump(best_estimator, model_filename)
        
# Predict on the test set using the best estimator
y_pred_test = np.abs(best_estimator.predict(X_test))

# Evaluate the model on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
print(f'Test RMSE: {rmse_test}')
print(f'Test MAE: {mae_test}')

r2_score_value = r2_score(y_test, y_pred_test)
print(f'R² score: {r2_score_value}')

# Update the dataframe with predictions from the best model
df['elapsed time:timestamp XGBoost'] = np.abs(np.where(best_estimator.predict(X) < 0, np.abs(best_estimator.predict(X)) / 10000, best_estimator.predict(X) / 1000))
df['elapsed time:timestamp'] = df['elapsed time:timestamp'].mask(df['elapsed time:timestamp'] < 0)
df = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'year', 'month', 'day', 'hour', 'elapsed time:timestamp - lag_10', 'elapsed time:timestamp - lag_9', 'elapsed time:timestamp - lag_8', 'elapsed time:timestamp - lag_7', 'elapsed time:timestamp - lag_6', 'elapsed time:timestamp - lag_5', 'elapsed time:timestamp - lag_4', 'elapsed time:timestamp - lag_3', 'elapsed time:timestamp - lag_2', 'elapsed time:timestamp - lag_1', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10'], errors='ignore')
df['elapsed time:timestamp'] = pd.to_timedelta(df['elapsed time:timestamp'], unit='seconds')
df['elapsed time:timestamp XGBoost'] = pd.to_timedelta(df['elapsed time:timestamp XGBoost'], unit='seconds')
df['next time:timestamp XGBoost'] = df['time:timestamp'] + df['elapsed time:timestamp XGBoost']
df = df.drop(columns=['elapsed time:timestamp XGBoost', 'elapsed time:timestamp'], errors='ignore')

# Decode categorical columns
for column in df.columns:
    if column in ['concept:name', 'concept:name - lag_1', 'concept:name - lag_2', 'concept:name - lag_3', 'concept:name - lag_4', 'concept:name - lag_5', 'concept:name - lag_6', 'concept:name - lag_7', 'concept:name - lag_8', 'concept:name - lag_9', 'concept:name - lag_10', 'next concept:name rfc', 'org:resource', 'Weekday', 'working_hours', 'vacation_day']:
        le = preprocessors[column]
        df[column] = le.inverse_transform(df[column])
    else:
        continue 

saver(df, f'data/{chosen_dataset}_rfc_xgboost.csv')

print_terminal_width_symbol('#')
print('\n')
print_centered_text('MODELS RESULTS')
print('\n')
print(f"""
      Next Event Prediction Accuracy (Random Forest Classifier):  {rfc_score}\n
      Next Time Prediction R\u00B2 score (XGBoost Regressor): {r2_score_value}\n
      RMSE Time Prediction (XGBoost): {rmse_test} in seconds\n
      MAE Time Prediction (XGBoost): {mae_test} in seconds\n
""")
print_terminal_width_symbol('#')
