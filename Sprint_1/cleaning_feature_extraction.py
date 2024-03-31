import pm4py
import os
import sys 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from func import saver, print_centered_text, print_terminal_width_symbol
from config import path_to_data_folder, slice_index, chosen_dataset

custom_format = "{desc}: {percentage:.0f}%\x1b[33m|\x1b[0m\x1b[32m{bar}\x1b[0m\x1b[31m{remaining}\x1b[0m\x1b[33m|\x1b[0m {n}/{total} [{elapsed}<{remaining}]"

# data folder creation
os.makedirs(f'{os.getcwd()}/data', exist_ok=True)
path_to_data_dick = f'{os.getcwd()}/data'

# artifacts folder creation
os.makedirs(f'{os.getcwd()}/artifacts_exp', exist_ok=True)
path_to_artifacts = f'{os.getcwd()}/artifacts_exp'

if not os.path.exists(f'{path_to_data_dick}/BPI_Challenge_2012.csv'):
    log_2012 = pm4py.read_xes(f'{path_to_data_folder}/BPI_Challenge_2012.xes.gz')
    pm4py.view_dotted_chart(log_2012, format="png")

    map_2012 = pm4py.discover_heuristics_net(log_2012)
    pm4py.view_heuristics_net(map_2012)

    dataframe_2012 = pm4py.convert_to_dataframe(log_2012)
    print(f"Extracted BPI_Challenge_2012.xes.gz\n")

    saver(dataframe_2012, f'{path_to_data_dick}/BPI_Challenge_2012.csv')
    print(f"\nStored BPI_Challenge_2012.csv\n")

if not os.path.exists(f'{path_to_data_dick}/BPI_Challenge_2017.csv'):
    log_2017 = pm4py.read_xes(f'{path_to_data_folder}/BPI Challenge 2017.xes.gz')
    pm4py.view_dotted_chart(log_2017, format="png")
    map_2017 = pm4py.discover_heuristics_net(log_2017)
    pm4py.view_heuristics_net(map_2017)

    dataframe_2017 = pm4py.convert_to_dataframe(log_2017)
    print(f"Extracted BPI Challenge 2017.xes.gz\n")
    saver(dataframe_2017, f'{path_to_data_dick}/BPI_Challenge_2017.csv')
    print(f"\nStored BPI_Challenge_2017.csv\n")

if not os.path.exists(f'{path_to_data_dick}/Road_Traffic_Fine_Management_Process.csv'):
    log_traffic = pm4py.read_xes(f'{path_to_data_folder}/Road_Traffic_Fine_Management_Process.xes.gz')
    pm4py.view_dotted_chart(log_traffic, format="png")
    map_traffic = pm4py.discover_heuristics_net(log_traffic)
    pm4py.view_heuristics_net(map_traffic)
    
    dataframe_traffic = pm4py.convert_to_dataframe(log_traffic)
    print(f"Extracted Road_Traffic_Fine_Management_Process.xes.gz\n")
    saver(dataframe_traffic, f'{path_to_data_dick}/Road_Traffic_Fine_Management_Process.csv')
    print(f"\nStored Road_Traffic_Fine_Management_Process.csv\n")

    print(f"All new files are stored in {path_to_data_dick}\n")


file_list = ['BPI_Challenge_2012', 'BPI_Challenge_2017', 'Road_Traffic_Fine_Management_Process']

# preprocess the dataframes in the same way
for file_name in file_list:
    if slice_index == None:
        df = pd.read_csv(f"{path_to_data_dick}/{file_name}.csv")
    else:
        df = pd.read_csv(f"{path_to_data_dick}/{file_name}.csv")[:slice_index]

    df = df.dropna(subset=['case:concept:name'])

    # convert them in integers
    if file_name == 'BPI_Challenge_2017':
        df['case:concept:name'] = df['case:concept:name'].str.split('_').str[1].astype(int)
    elif file_name == 'BPI_Challenge_2012':
        df['case:concept:name'] = df['case:concept:name'].astype(int)
    elif file_name == 'Road_Traffic_Fine_Management_Process':
        df['case:concept:name'] = df['case:concept:name'].str.extract(r'(\d+)').astype(int)
    else:
        # Try converting the column to numeric
        df['case:concept:name'] = pd.to_numeric(df['case:concept:name'], errors='coerce')

        # Check for non-numeric values (NaN after coercion)
        non_numeric_values = df[df['case:concept:name'].isna()]

        if non_numeric_values.empty:
            print("All values in the case:concept:name are numeric.")
        else:
            print("Non-numeric values found in case:concept:name:", non_numeric_values)

    # sort the df according to the ids
    df = df.sort_values(by='time:timestamp', ascending=True)

    # remove unnamed column
    df = df.drop(columns=['Unnamed: 0'])

    # deal with timestamps 
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
    df = df.dropna(subset=['time:timestamp'])
    df['time:timestamp'] = df['time:timestamp'].dt.strftime('%d:%m:%Y %H:%M:%S.%f')

    # Calculate the time difference between consecutive rows
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%d:%m:%Y %H:%M:%S.%f', errors='coerce')
    df['timestamp_difference'] = df.groupby('case:concept:name')['time:timestamp'].diff()

    if file_name == 'BPI_Challenge_2012':
        df['case:REG_DATE'] = pd.to_datetime(df['case:REG_DATE'], errors='coerce')
        df = df.dropna(subset=['case:REG_DATE'])
        df['case:REG_DATE'] = df['case:REG_DATE'].dt.strftime('%d:%m:%Y %H:%M:%S.%f')

    if file_name == 'BPI_Challenge_2012':
        df_2012 = df
        print(f"{file_name} cleaning is finished\n")
        saver(df_2012, f"{path_to_data_dick}/clean_BPI_Challenge_2012.csv")
        print(f"\n{file_name} is stored in clean format\n")

    if file_name == 'BPI_Challenge_2017':
        df_2017 = df
        print(f"{file_name} cleaning is finished\n")
        saver(df_2017, f"{path_to_data_dick}/clean_BPI_Challenge_2017.csv")
        print(f"\n{file_name} is stored in clean format\n")

    if file_name == 'Road_Traffic_Fine_Management_Process':
        df_road_traffic = df
        print(f"{file_name} cleaning is finished\n")
        saver(df_road_traffic, f"{path_to_data_dick}/clean_Road_Traffic_Fine_Management_Process.csv")
        print(f"\n{file_name} is stored in clean format\n")


df_2017_clean = pd.read_csv(f'{path_to_data_dick}/clean_BPI_Challenge_2017.csv')
df_2012_clean = pd.read_csv(f'{path_to_data_dick}/clean_BPI_Challenge_2012.csv')
df_road_traffic_clean = pd.read_csv(f'{path_to_data_dick}/clean_Road_Traffic_Fine_Management_Process.csv', low_memory=False)

df_2017_clean[['start_timestamp', "time:timestamp"]]
df_2017_clean['timestamp_date'] = pd.to_datetime(df_2017_clean['time:timestamp'])
df_2017_clean['day_of_the_week'] = df_2017_clean['timestamp_date'].dt.day_name()

df_2012_clean[['start_timestamp', "time:timestamp"]]
df_2012_clean['timestamp_date'] = pd.to_datetime(df_2012_clean['time:timestamp'])
df_2012_clean['day_of_the_week'] = df_2012_clean['timestamp_date'].dt.day_name()

df_road_traffic_clean[['start_timestamp', "time:timestamp"]]
df_road_traffic_clean['timestamp_date'] = pd.to_datetime(df_road_traffic_clean['time:timestamp'])
df_road_traffic_clean['day_of_the_week'] = df_road_traffic_clean['timestamp_date'].dt.day_name()


df_2017_clean['day_of_the_week']
weekday_list = []
for i in df_2017_clean['day_of_the_week']:
    if i in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        weekday_list.append(1)
    else:
        weekday_list.append(0)
df_2017_clean['Weekday'] = weekday_list

df_2012_clean['day_of_the_week']
weekday_list = []
for i in df_2012_clean['day_of_the_week']:
    if i in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        weekday_list.append(1)
    else:
        weekday_list.append(0)
df_2012_clean['Weekday'] = weekday_list

df_road_traffic_clean['day_of_the_week']
weekday_list = []
for i in df_road_traffic_clean['day_of_the_week']:
    if i in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        weekday_list.append(1)
    else:
        weekday_list.append(0)
df_road_traffic_clean['Weekday'] = weekday_list


# Add a new column indicating if the timestamp is within the specified time range
df_2017_clean['working_hours'] = ((df_2017_clean['timestamp_date'].dt.time >= pd.to_datetime('09:30').time()) & (df_2017_clean['timestamp_date'].dt.time <= pd.to_datetime('16:30').time())).astype(int)
df_2012_clean['working_hours'] = ((df_2012_clean['timestamp_date'].dt.time >= pd.to_datetime('09:30').time()) & (df_2012_clean['timestamp_date'].dt.time <= pd.to_datetime('16:30').time())).astype(int)
df_road_traffic_clean['working_hours'] = ((df_road_traffic_clean['timestamp_date'].dt.time >= pd.to_datetime('09:30').time()) & (df_road_traffic_clean['timestamp_date'].dt.time <= pd.to_datetime('16:30').time())).astype(int)

vacation_days = [
    pd.Timestamp('2016-01-01'), pd.Timestamp('2016-01-06'), pd.Timestamp('2016-02-07'),
    pd.Timestamp('2016-02-14'), pd.Timestamp('2016-03-25'), pd.Timestamp('2016-03-27'),
    pd.Timestamp('2016-03-28'), pd.Timestamp('2016-03-29'), pd.Timestamp('2016-04-27'),
    pd.Timestamp('2016-05-01'), pd.Timestamp('2016-05-04'), pd.Timestamp('2016-05-08'),
    pd.Timestamp('2016-05-15'), pd.Timestamp('2016-05-26'), pd.Timestamp('2016-05-28'),
    pd.Timestamp('2016-05-29'), pd.Timestamp('2016-06-19'), pd.Timestamp('2016-09-20'),
    pd.Timestamp('2016-10-04'), pd.Timestamp('2016-10-30'), pd.Timestamp('2016-10-31'),
    pd.Timestamp('2016-11-11'), pd.Timestamp('2016-12-05'), pd.Timestamp('2016-12-25'),
    pd.Timestamp('2016-12-26'), pd.Timestamp('2016-12-31'), pd.Timestamp('2017-01-01'),
    pd.Timestamp('2017-02-14'), pd.Timestamp('2017-02-26'), pd.Timestamp('2017-03-26'),
    pd.Timestamp('2017-04-14'), pd.Timestamp('2017-04-16'), pd.Timestamp('2017-04-17'),
    pd.Timestamp('2017-04-27'), pd.Timestamp('2017-05-01'), pd.Timestamp('2017-05-04'),
    pd.Timestamp('2017-05-05'), pd.Timestamp('2017-05-14'), pd.Timestamp('2017-05-25'),
    pd.Timestamp('2017-05-27'), pd.Timestamp('2017-06-04'), pd.Timestamp('2017-06-05'),
    pd.Timestamp('2017-06-18'), pd.Timestamp('2017-06-25'), pd.Timestamp('2017-09-01'),
    pd.Timestamp('2017-09-19'), pd.Timestamp('2017-10-04'), pd.Timestamp('2017-10-29'),
    pd.Timestamp('2017-10-31'), pd.Timestamp('2017-11-11'), pd.Timestamp('2017-12-05'),
    pd.Timestamp('2017-12-25'), pd.Timestamp('2017-12-26'), pd.Timestamp('2017-12-31')
]
new_list = []
for i,x in enumerate(vacation_days):
    for i_2,v in enumerate(df_2017_clean['timestamp_date'].dt.date):
        if i == 0:
            if x.date() == v:
                new_list.append(1)
            else:
                new_list.append(0)
        else:
            if x.date() == v:
                new_list[i_2] = 1

df_2017_clean['vacation_day'] = new_list

vacation_days = [
    pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-06'), pd.Timestamp('2011-02-14'),
    pd.Timestamp('2011-03-06'), pd.Timestamp('2011-03-27'), pd.Timestamp('2011-04-22'),
    pd.Timestamp('2011-04-24'), pd.Timestamp('2011-04-25'), pd.Timestamp('2011-04-30'),
    pd.Timestamp('2011-05-01'), pd.Timestamp('2011-05-04'), pd.Timestamp('2011-05-05'),
    pd.Timestamp('2011-05-08'), pd.Timestamp('2011-06-02'), pd.Timestamp('2011-06-12'),
    pd.Timestamp('2011-06-13'), pd.Timestamp('2011-06-19'), pd.Timestamp('2011-09-20'),
    pd.Timestamp('2011-10-04'), pd.Timestamp('2011-10-30'), pd.Timestamp('2011-10-31'),
    pd.Timestamp('2011-11-11'), pd.Timestamp('2011-12-05'), pd.Timestamp('2011-12-25'),
    pd.Timestamp('2011-12-26'), pd.Timestamp('2011-12-31'), pd.Timestamp('2012-01-01'),
    pd.Timestamp('2012-02-19'), pd.Timestamp('2012-03-25'), pd.Timestamp('2012-04-06'),
    pd.Timestamp('2012-04-08'), pd.Timestamp('2012-04-09'), pd.Timestamp('2012-05-13'),
    pd.Timestamp('2012-05-17'), pd.Timestamp('2012-05-27'), pd.Timestamp('2012-05-28'),
    pd.Timestamp('2012-06-17'), pd.Timestamp('2012-09-18'), pd.Timestamp('2012-10-28'),
    pd.Timestamp('2012-12-25'), pd.Timestamp('2012-12-26')
]
new_list = []
for i,x in enumerate(vacation_days):
    for i_2,v in enumerate(df_2012_clean['timestamp_date'].dt.date):
        if i == 0:
            if x.date() == v:
                new_list.append(1)
            else:
                new_list.append(0)
        else:
            if x.date() == v:
                new_list[i_2] = 1

df_2012_clean['vacation_day'] = new_list


file_list = ['BPI_Challenge_2012', 'BPI_Challenge_2017', 'Road_Traffic_Fine_Management_Process']

for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        print(f"{file_name} is finished\n")
        saver(df_2012_clean, f"{path_to_data_dick}/clean_BPI_Challenge_2012.csv")
        print(f"\n{file_name} is stored in clean format with features\n")

    if file_name == 'BPI_Challenge_2017':
        print(f"{file_name} is finished\n")
        saver(df_2017_clean, f"{path_to_data_dick}/clean_BPI_Challenge_2017.csv")
        print(f"\n{file_name} is stored in clean format with features\n")

    if file_name == 'Road_Traffic_Fine_Management_Process':
        print(f"{file_name} is finished\n")
        saver(df_road_traffic_clean, f"{path_to_data_dick}/clean_Road_Traffic_Fine_Management_Process.csv")
        print(f"\n{file_name} is stored in clean format with features\n")


print_terminal_width_symbol("#")
print("\n")
print_centered_text("VISUALIZATION OF DATA AFTER PREPROCESSING")
print("\n")

        
df_2012_clean['time_difference'] = [pd.to_timedelta(td).total_seconds() for td in df_2012_clean['timestamp_difference']]
df_2017_clean['time_difference'] = [pd.to_timedelta(td).total_seconds() for td in df_2017_clean['timestamp_difference']]
df_road_traffic_clean['time_difference'] = [pd.to_timedelta(td).total_seconds() for td in df_road_traffic_clean['timestamp_difference']]

for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        pivot_df = df_2012_clean.pivot_table(index='concept:name', columns='vacation_day', values='time_difference', aggfunc='mean')
        pivot_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and Vacation Day')
        plt.xlabel('Concept Name')
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Vacation Day', labels=['Not Vacation', 'Vacation'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'BPI_Challenge_2017':
        pivot_df = df_2017_clean.pivot_table(index='concept:name', columns='vacation_day', values='time_difference', aggfunc='mean')
        pivot_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and Vacation Day')
        plt.xlabel('Concept Name')
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Vacation Day', labels=['Not Vacation', 'Vacation'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))


for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        pivot_df_2 = df_2012_clean.pivot_table(index='concept:name', columns='Weekday', values='time_difference', aggfunc='mean')
        pivot_df_2.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and Weekday')
        plt.xlabel('Concept Name')
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Week Day', labels=['Not a Weekday', 'Weekday'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'BPI_Challenge_2017':
        pivot_df_2 = df_2017_clean.pivot_table(index='concept:name', columns='Weekday', values='time_difference', aggfunc='mean')
        pivot_df_2.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and Weekday')
        plt.xlabel('Concept Name')
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Week Day', labels=['Not a Weekday', 'Weekday'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))
    
    if file_name == 'Road_Traffic_Fine_Management_Process':
        pivot_df_2 = df_road_traffic_clean.pivot_table(index='concept:name', columns='Weekday', values='time_difference', aggfunc='mean')
        pivot_df_2.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and Weekday')
        plt.xlabel('Concept Name')
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Week Day', labels=['Not a Weekday', 'Weekday'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        pivot_df_3 = df_2012_clean.pivot_table(index='concept:name', columns='working_hours', values='time_difference', aggfunc='mean')
        pivot_df_3.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and working hours')
        plt.xlabel('Concept Name')  
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Working Hours', labels=['Not working hours', 'working hours'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'BPI_Challenge_2017':
        pivot_df_3 = df_2017_clean.pivot_table(index='concept:name', columns='working_hours', values='time_difference', aggfunc='mean')
        pivot_df_3.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and working hours')
        plt.xlabel('Concept Name')  
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Working Hours', labels=['Not working hours', 'working hours'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))
    
    if file_name == 'Road_Traffic_Fine_Management_Process':
        pivot_df_3 = df_road_traffic_clean.pivot_table(index='concept:name', columns='working_hours', values='time_difference', aggfunc='mean')
        pivot_df_3.plot(kind='bar', figsize=(10, 6))
        plt.title('Mean Time Difference per Concept Name and working hours')
        plt.xlabel('Concept Name')  
        plt.ylabel('Mean Time Difference (seconds)')
        plt.xticks(rotation=90)  # Rotate concept names vertically
        plt.legend(title='Working Hours', labels=['Not working hours', 'working hours'])
        plt.grid(True)
        plt.yscale('log')  # Set y-axis to log scale
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))


df_2012_clean_sequence = df_2012_clean
df_2017_clean_sequence = df_2017_clean
df_road_traffic_clean_sequence = df_road_traffic_clean

for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        last_concept_per_group = df_2012_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        # Count the frequency of each 'concept:name' being the last in its group
        last_concept_freq = last_concept_per_group['last_concept'].value_counts()
        plt.figure(figsize=(10, 6))
        last_concept_freq.plot(kind='bar')
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title('Frequency of Concept Names Being the Last (Log Scale)')
        plt.xlabel('Concept Name')
        plt.ylabel('Frequency (log scale)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'BPI_Challenge_2017':
        last_concept_per_group = df_2017_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        # Count the frequency of each 'concept:name' being the last in its group
        last_concept_freq = last_concept_per_group['last_concept'].value_counts()
        plt.figure(figsize=(10, 6))
        last_concept_freq.plot(kind='bar')
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title('Frequency of Concept Names Being the Last (Log Scale)')
        plt.xlabel('Concept Name')
        plt.ylabel('Frequency (log scale)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'Road_Traffic_Fine_Management_Process':
        last_concept_per_group = df_road_traffic_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        #Count the frequency of each 'concept:name' being the last in its group
        last_concept_freq = last_concept_per_group['last_concept'].value_counts()
        plt.figure(figsize=(10, 6))
        last_concept_freq.plot(kind='bar')
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title('Frequency of Concept Names Being the Last (Log Scale)')
        plt.xlabel('Concept Name')
        plt.ylabel('Frequency (log scale)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

for file_name in file_list:
    if file_name == 'BPI_Challenge_2012':
        mean_requested_amount_per_group = df_2012_clean_sequence.groupby('case:concept:name')['case:AMOUNT_REQ'].mean().reset_index(name='mean_requested_amount')
        # Group the data by 'case:concept:name' and get the last concept of each sequence
        last_concept_per_group = df_2012_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        # Merge the mean requested amount data with the last concept data
        df_merged = pd.merge(last_concept_per_group, mean_requested_amount_per_group, on='case:concept:name')
        # Create a scatter plot to visualize the relationship between the requested amount and the last concept of the sequence
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_merged, x='mean_requested_amount', y='last_concept')
        plt.title('Relationship between Requested Amount and Last Concept of Sequence')
        plt.xlabel('Requested Amount')
        plt.ylabel('Last Concept')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'BPI_Challenge_2017':
        mean_requested_amount_per_group = df_2017_clean_sequence.groupby('case:concept:name')['case:RequestedAmount'].mean().reset_index(name='mean_requested_amount')
        # Group the data by 'case:concept:name' and get the last concept of each sequence
        last_concept_per_group = df_2017_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        # Merge the mean requested amount data with the last concept data
        df_merged = pd.merge(last_concept_per_group, mean_requested_amount_per_group, on='case:concept:name')
        # Create a scatter plot to visualize the relationship between the requested amount and the last concept of the sequence
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_merged, x='mean_requested_amount', y='last_concept')
        plt.title('Relationship between Requested Amount and Last Concept of Sequence')
        plt.xlabel('Requested Amount')
        plt.ylabel('Last Concept')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

    if file_name == 'Road_Traffic_Fine_Management_Process':
        mean_requested_amount_per_group = df_road_traffic_clean_sequence.groupby('case:concept:name')['totalPaymentAmount'].mean().reset_index(name='mean_requested_amount')
        # Group the data by 'case:concept:name' and get the last concept of each sequence
        last_concept_per_group = df_road_traffic_clean_sequence.groupby('case:concept:name')['concept:name'].last().reset_index(name='last_concept')
        # Merge the mean requested amount data with the last concept data
        df_merged = pd.merge(last_concept_per_group, mean_requested_amount_per_group, on='case:concept:name')
        # Create a scatter plot to visualize the relationship between the requested amount and the last concept of the sequence
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_merged, x='mean_requested_amount', y='last_concept')
        plt.title('Relationship between Total Payment Amount and Last Concept of Sequence')
        plt.xlabel('Total Payment Amount')
        plt.ylabel('Last Concept')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

for file_name in file_list:
    if file_name == 'BPI_Challenge_2017':
        loan_application_steps = df_2017_clean_sequence.groupby('case:concept:name').size().reset_index(name='step_count')
        # Determine if a credit score is present for each loan application
        # Assuming 'credit_score' is the column indicating the presence of a credit score
        loan_application_steps['credit_score_present'] = df_2017_clean_sequence.groupby('case:concept:name')['CreditScore'].any().astype(int).reset_index(drop=True)
        # Plot the distribution of step counts for cases with and without a credit score
        plt.figure(figsize=(10, 6))
        plt.hist(loan_application_steps[loan_application_steps['credit_score_present'] == 0]['step_count'], bins=20, alpha=0.5, label='No Credit Score')
        plt.hist(loan_application_steps[loan_application_steps['credit_score_present'] == 1]['step_count'], bins=20, alpha=0.5, label='Credit Score Present')
        plt.title('Distribution of Step Counts by Credit Score Presence')
        plt.xlabel('Number of Steps')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(path_to_artifacts, f"{plt.gca().get_title()}.png"))

print_terminal_width_symbol('#')
print('\n')
print_centered_text("EXTRACTION OF TRACES AND SAVING THEM FOR THE MODEL")

if not os.path.exists(f'data/traces_{chosen_dataset}.json'):
    dataframe_train = pd.read_csv(f'data/{chosen_dataset}.csv')

    dataframe_train = pm4py.format_dataframe(dataframe_train, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

    event_log_train = pm4py.convert_to_event_log(dataframe_train)
        
    pm4py.write_xes(event_log_train, f'data/event_log_{chosen_dataset}.xes')

    event_logs_files_names = [f'event_log_{chosen_dataset}']

    for event_log_file in event_logs_files_names:

        log = pm4py.read_xes(os.path.join(f"data/{event_log_file}.xes"))
        log = pm4py.convert_to_event_log(log)

        trace_df_list = []
        for trace in tqdm(log, desc="Processing traces", dynamic_ncols=True, bar_format=custom_format, ascii=' -'):
            case_concept_name = trace.attributes['concept:name']
            concept_name_col = []
            timestamp_col = []
            
            for event in trace:
                concept_name_col.append(event['concept:name'])
                timestamp_col.append(str(event['time:timestamp']))
            
            df_trace = pd.DataFrame({'concept:name': concept_name_col, 'time:timestamp': timestamp_col})
            df_trace = pd.concat([df_trace, pd.DataFrame({'concept:name': ['END'], 'time:timestamp': [timestamp_col[-1]]})], ignore_index=True)
            df_trace.name = f'{case_concept_name}'

            trace_df_list.append(df_trace)

        df_final = pd.DataFrame({'case:concept:name': [df.name for df in trace_df_list], 'trace': trace_df_list})

        df_final.to_json(f'data/traces_{chosen_dataset}.json', orient='records')

        if os.path.exists(f'data/{event_log_file}.xes'):
            # Remove the file
            os.remove(f'data/{event_log_file}.xes')

print(f'Traces are extracted and saved in the data folder')