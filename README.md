# Process Mining Project

Who has not experienced the situation that you submitted a request into some administrative system (like a reimbursement form, a purchase of a ticket in a webshop, an insurance claim or a request in one of TU/e's administrative systems). How often have you wondered: How long will this request take? 

In this Project, we look at this problem from a process mining perspective. We use data coming from such a system to to develop a prediction model to predict (a) the next step/activity in the process and (b) when that next activity will happen. After having developed one (or two) prediction models that predict the next step and time until that next step, we should generalize the model to be capable of (c) predicting the entire continuation or suffix (both activities and time) of an ongoing process execution until completion.

This prediction model is to be developed using Python using whatever data analysis techniques the members in this group think of.

In general, process prediction techniques can be categorized into local and global techniques. Local techniques use information from the current case (for example the insurance claim) only to base the prediction on. Global techniques use all available information, such as for example the load of the system or even today's weather in their predictions. 
___
## The Challenge 

The challenge in this project is to not only develop a tool to do the predictions, but to also carefully think about the context in which these predictions are made and the implications that this has for feature selection and quantitative and qualitative evaluation of the prediction results. There are some essential differences between process mining and regular data mining activities and these will become apparent over this project.
___

## Files

All files are stored in the following way:
- **Sprint 1 folder** - files being written in the first sprint
- **Sprint 2 folder** - files being written in the second sprint
- **Sprint 3 folder** - files being written in the third sprint

- **requirements.txt** - this file includes all required libraries that you need to have to run the code.
- **config.py** - this file includes global variables that you have to configure for your local environment
- **main.py** - this file you have to run to run the whole program

- **Sprint 1 folder** :
   - **cleaning_feature_extraction.py** - this file includes the whole code for cleaning of the data, extraction of features from the data and at the same time saving all the data in the data folder in your environment
   - **data_exp_vis.ipynb** - visualizations of the extracted data
   - **train_test_sep.ipynb** - code for demo representation of how the train and test separations is done 
   - **visualizations.ipynb** - visualizations of the data

- **Sprint 2 folder** :
   - **models_run.py** - file which runs and assembles the models 
   - **train_test_split.py** - file with train test split function which is used during the project
   - **pm4py_work.ipynb** - file used for extra visualizations of the log files
   - **visualizations.ipynb** - file with visualizations for sprint 2

- **Sprint 3 folder**:
   - **trace_prediction.py** - file for running and assembling the recurrent model that predicts the traces
   - **traces_Seq2Seq.ipynb** - file with seq2seq model but weren't used in the project due to high complexity of the model and computational difficulties.
___
## The Data

The dataset for which we would like to predict the next event and time until next event can be any of the following datasets (in order of complexity):

- BPI Challenge 2012: [can be found here](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- Italian Road fines data: [can be found here](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- BPI Challenge 2017: [can be found here](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)
- BPI Challenge 2018: [can be found here](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)

To test and train our method, we are going to use synthetic datasets found [here](https://data.4tu.nl/search?q=:keyword:%20%22real%20life%20event%20logs%22)

___
## The datasets that mainly use in the explorations.
   Mainly most of the focus is put on the BPI_Challenge_2017, but the same time the work is not hard coded, and can be used for BPI_Challenge_2012 as well and for Traffic dataset at the same time. 
   Main assumption is that in the dataset we have columns like case:concept:name, concept:name and time:timestamp.

___
## The Deliverables  

A rough schedule for the sprint deliverables is as follows:

After two weeks, we should be able to read an event log into memory and to export a CSV file with an additional column showing the estimated time until next event (naive estimator) for the case to which the event belongs.

The import should be independent of the file and able to read all five types of columns: Literals (strings), Continuous (double), Discrete (long), Timestamp (Date in the format dd : MM : yyyy HH : mm : ss.SSS), Boolean (boolean in the format true/false). 

The export should include two additional columns specifying (1) the predicted next activity according to the naive predictor explained above and (2) the predicted time of that activity.

After four weeks, we should deliver the same output, but now with more additional columns, both for the naive estimator and for a new estimator we developed. Furthermore, we should show a first draft of a poster explaining how you build your new estimator and how it compares to the naive one. It is expected that we properly evaluate our model on an independent test set (On Training/Test (and Validation) data.).

After six weeks, again an updated tool is expected which can produce for any running process trace (prefix of a full trace) the suffix of activities and timestamps until completion of that trace. 

The final deliverable is the final poster (and the presentation thereof) about our work and the tool itself.

___
## How to set up environment

1. Download all files and place in one folder
2. Create virtual environment .venv in this folder (can be .conda if you want)
   
   For windows:
   ```python
   python -m venv .venv
   ```
   For Mac and Linux:
   ```python
   python3 -m venv .venv
   ```
3. Activate the virtual environment
   
   For Widows:
   ```python
   myenv\Scripts\activate
   ```
   For Mac and Linux:
   ```python
   source myenv/bin/activate
   ```
4. install all libraries that we used to develop this code
  ```python
  pip install -r requirements.txt
  ```
___
## How to run the code
1. Change the path to the original data in the **config.py** file, as well we recommend to slice the data, variable for this is also available in config file. 
2. Run the **main.py** file. All the progress can be seen in your terminal during the run process.