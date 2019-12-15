# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [Instructions](#instructions)

## Installation <a name="installation"></a>

To run his project, the following packages need to be installed for nltk:
* punkt
* wordnet
* stopwords

Also make sure that you have already installed package sklearn and sqlalchemy.
If it is not installed yet, you can simply run 'pip install sklearn' and 'pip install sqlalchemy' in your terminal 

## Project Motivation<a name="motivation"></a>

This project is intended to make a classification of messages that people sent during disaster. By sorting the messages received into specific category it will speed up the response process, it can be aid related, wheater info, or rescue.
To complete this project I applied data engineering, natural language processing, and machine learning skills.

## File Descriptions <a name="files"></a>

There are three main folders:
1. data
    - disaster_categories.csv: dataset including all the disaster categories e.g aid_center, wheater_related, rescue
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web applicatin

## Results<a name="results"></a>

1. An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
2. A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
3. A Flask app was created to show data visualization and classify the message that user enters on the web page.

Below is screenshot of the disaster aplication of overview datasets:
![alt text](https://github.com/dhanys24/disaster-response-pipeline/blob/master/app/overview_dataset.png)


And here is the application that need machine learning to specify category of the received messages.
In the screenshot below, the example of the messages is "Please, we need tents and water. We are in Silo, Thank you!".
And the category detected are Related, Request, Aid related, Water, and Shelter.
![alt text](https://github.com/dhanys24/disaster-response-pipeline/blob/master/app/messages_categories_result.png)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project. 
## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline which read data from 2 csv files and store it into database 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline which trains classifier and save the model as pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
