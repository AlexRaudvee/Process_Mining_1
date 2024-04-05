import subprocess
from config import first_run

if first_run:
    # Call cleaning_feature_extraction.py 
    subprocess.run(['python', 'Sprint_1/cleaning_feature_extraction.py'])
    """
    This file is going to convert xes in csv files and with primary filtering and cleaning.
    In addition, it will extract the traces from the csv files.
    """

# Call models_run.py 
subprocess.run(['python', 'Sprint_2/models_run.py'])
"""
This file trains and evaluates the models and makes predictions on the datasets for next 
event predictions and next time prediction.
"""

# Call trace_prediction.py
subprocess.run(['python', 'Sprint_3/trace_prediction.py'])
"""
This file predicts the suffixes of the traces by using above trained models and saves them
in the separate csv file
"""