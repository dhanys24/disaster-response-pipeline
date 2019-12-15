# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import sys
import pickle

def load_data(database_filepath):
    '''
    Load datasets from dataframe
    INPUT :
        database_filepath - a SQLite database path
    OUTPUT :
        X - messages (features)
        Y - categories (target)
        category_names - labels for 36 categories
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages',engine) 
    X = df.message.values
    Y = (df[df.columns[4:]]).values
    category_names = list(df.columns[4:])

    return X,Y,category_names

def tokenize(text):
    '''
    Tokenize and clean text
    INPUT:
        text - original message text
    OUTPUT:
        lemmed - Tokenized, cleaned, and lemmatized text
    '''
    
    # NORMALIZE case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # TOKENIZE text
    tokens = word_tokenize(text)
    
    # LEMMATIZE andremove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word,pos='n') for word in tokens if word not in stop_words]

    return tokens

def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    
    INPUT: None
    OUTPUT:
        Results of GridSearchCV
    '''
    
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    parameters = {'clf__estimator__n_estimators': [100,200], # by default 100
                  'clf__estimator__min_samples_split': [2,3], # by default 2
                  'clf__estimator__criterion': ['entropy', 'gini'] # by default gini
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    INPUT: 
        model -  Model to be evaluated
        X_test - Test data (features)
        Y_test - True lables for Test data
        category_names - Labels for 36 categories
    OUTPUT:
        Print accuracy and classfication report for each category
    '''
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
    
        # Calculate the accuracy for each of categories.
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test[:, i], Y_pred[:,i])))
    
def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    INPUT: 
        model - Model to be saved
        model_filepath - path of the output pick file
    OUTPUT:
        A pickle file of saved model
    '''
    pickle.dump(model,open(model_filepath, "wb"))
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        
if __name__ == '__main__':
    main()
