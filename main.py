import subprocess

# Call cleaning_feature_extraction.py 
subprocess.run(['python', 'Sprint_1/cleaning_feature_extraction.py'])

# Call models_run.py 
subprocess.run(['python', 'Sprint_2/models_run.py'])

# Call trace_prediction.py
subprocess.run(['python', 'Sprint_3/trace_prediction.py'])