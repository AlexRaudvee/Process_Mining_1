# Process Mining Project

Who has not experienced the situation that you submitted a request into some administrative system (like a reimbursement form, a purchase of a ticket in a webshop, an insurance claim or a request in one of TU/e's administrative systems). How often have you wondered: How long will this request take? 

In this Project, we look at this problem from a process mining perspective. We use data coming from such a system to to develop a prediction model to predict (a) the next step/activity in the process and (b) when that next activity will happen. After having developed one (or two) prediction models that predict the next step and time until that next step, we should generalize the model to be capable of (c) predicting the entire continuation or suffix (both activities and time) of an ongoing process execution until completion.

This prediction model is to be developed using Python using whatever data analysis techniques the members in this group think of.

In general, process prediction techniques can be categorized into local and global techniques. Local techniques use information from the current case (for example the insurance claim) only to base the prediction on. Global techniques use all available information, such as for example the load of the system or even today's weather in their predictions. 

## The Challenge

The challenge in this project is to not only develop a tool to do the predictions, but to also carefully think about the context in which these predictions are made and the implications that this has for feature selection and quantitative and qualitative evaluation of the prediction results. There are some essential differences between process mining and regular data mining activities and these will become apparent over this project.

 Files

- **requirements.txt** - this file includes all required libraries that you need to have to run the code.
- **advanced_models_2012_2017.ipynb.ipynb** - here you can find the additional things for extracting the data from naive model.
- **data_cleaning_2012_2017.ipynb** - this file shows how the cleaning of the data were done before applying the naive model and some of the visualizations
- **naive_model.ipynb** - this file represents the testing of the naive model, and small visualizations of dataset including the evaluation of the model for 'BPI_Challenge_2017'.
- **train_test_sep.ipynb** - this is for splitting the training and testing data with the ratio 8:2 for 'BPI_Challenge_2017'.
- **cleaning_feature_extraction.py** - this file includes some of the cleaning methods and find out the feature extactions for 'BPI_Challenge_2017', 'BPI Challenge 2012' and 'Italian Road fines data'.
- **data_exp_vis.ipynb** - this file includes the first visualizations for the first and primary data exploration
- **visualizations.ipynb** and **visualization.ipynb** - in these files you can find all the main visualizatons that were done for the sprints.
- **data_cleaning_2012_2017_start.ipynb** - in this file, using the cleaned dataset for 2017, and making some visualizations about 'concept: name'.
- **models_run.py** - this file is for the models results about naive model, XGBoost and RFC.
- **modesl_train_xgboost_rfc.ipynb** - this file is using train and testing splitting datasets to get the models results about XGBoost and RFC.
- **pm4py_work.ipynb** - this file is for finding the process tress about 'BPI_Challenge_2017'.
- **traces_extraction.ipynb** - this file finds the actual traces based on 'case: concept: name' with 'BPI_Challenge_2017' dataset.
- **train_test_split.py** - this file is for the functions of how to split the train and test dataset.


## The Data

The dataset for which we would like to predict the next event and time until next event can be any of the following datasets (in order of complexity):

- BPI Challenge 2012: [can be found here](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- Italian Road fines data: [can be found here](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- BPI Challenge 2017: [can be found here](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)
- BPI Challenge 2018: [can be found here](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)

To test and train our method, we are going to use synthetic datasets found [here](https://data.4tu.nl/search?q=:keyword:%20%22real%20life%20event%20logs%22)

## The datasets that mainly use in the explorations.
- **data_cleaning_2012_2017.ipynb**: running this for the 'clean_BPI_Challenge_2017'
- **train_test_sep.ipynb**: using 'clean_BPI_Challenge_2012' as input dataset to get 3 separate files, testing, training and the merged dataset.


## The Delivarables 

A rough schedule for the sprint deliverables is as follows:

After two weeks, we should be able to read an event log into memory and to export a CSV file with an additional column showing the estimated time until next event (naive estimator) for the case to which the event belongs.

The import should be independent of the file and able to read all five types of columns: Literals (strings), Continuous (double), Discrete (long), Timestamp (Date in the format dd : MM : yyyy HH : mm : ss.SSS), Boolean (boolean in the format true/false). 

The export should include two additional columns specifying (1) the predicted next activity according to the naive predictor explained above and (2) the predicted time of that activity.

After four weeks, we should deliver the same output, but now with more additional columns, both for the naive estimator and for a new estimator we developed. Furthermore, we should show a first draft of a poster explaining how you build your new estimator and how it compares to the naive one. It is expected that we properly evaluate our model on an independent test set (On Training/Test (and Validation) data.).

After six weeks, again an updated tool is expected which can produce for any running process trace (prefix of a full trace) the suffix of activities and timestamps until completion of that trace. 

The final deliverable is the final poster (and the presentation thereof) about our work and the tool itself.

## How to set up enviorment

1. Download all files and place in one folder
2. Create virtual envioremnt .venv in this folder (can be .conda if you want)
   
   For windows:
   ```python
   python -m venv .venv
   ```
   For Mac and Linux:
   ```python
   python3 -m venv .venv
   ```
3. Activate the virtual enviroment
   
   For Windovs:
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

## How to run the code

1. First you will have to change the variables in config.py (description of variables inside the file)
2. Then you run cleaning_feature_extraction.py this code will create a data folder with all needed inputs for other files (this code prints out images, so close them for continue of the code running)
3. For obtaining results from the models, you will have to run models_run.py, this file will print out results and graphs (close the graphs to continue running the code)
4. trace_prediction.ipynb is still in progress as this model is quiete tough you can run and obtain the results as a dataframe of one of the trace or for all of them (can take a long time)
5. visualization files can be run as well, but most of the important images are returning during the run of the above mentioned files.
