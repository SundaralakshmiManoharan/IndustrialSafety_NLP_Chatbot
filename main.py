# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# required library



import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS
from plotly.subplots import make_subplots
from  nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from IPython.core.display import HTML
import   plotly.graph_objects as go
from  nltk.corpus import stopwords
from sklearn.utils import resample
from IPython.display import Image
import matplotlib.pyplot as plt
from nltk import tokenize,stem
import plotly.express as px
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import unicodedata
import numpy as np
import stylecloud
import plotly
import string
import nltk
import re
import os
from PIL import Image
import streamlit.components.v1 as components
import webbrowser
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC
from keras.layers import Embedding, Dense, Dropout, Input, LSTM, GlobalMaxPool1D, BatchNormalization, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras import regularizers, optimizers
from keras.layers import TextVectorization
from keras.metrics import binary_accuracy
from keras.initializers import Constant
from keras.models import Sequential
from tensorflow import keras
from keras import backend
import tensorflow as tf
import spacy.cli
import spacy 
from keras.models import load_model
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer 
from streamlit_chat import message 
import json
from streamlit_option_menu import option_menu
from spacy.lang.en.examples import sentences 



from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score
)





# Setting the image - 
image = Image.open('images/chta-bot_Blog_3.png')

# Setting the image width -
st.image(image, use_column_width=True)


# Sidebar navigation for users -
#st.sidebar.header('Navigation tab -')
#navigation_tab = st.sidebar.selectbox('Choose a tab', ('Home-Page', 'Exploratory Data Analysis', 'NLP Text Processing', 'Machine Learning Classifiers'))

  

ProjectInformation = st.container()
data = st.container()
EDA = st.container()
NLP = st.container()
download = st.container()
dataset = st.container()
features = st.container()
machinelearning_model_training = st.container()
NN_model_training = st.container()
LSTM_model_training = st.container()
classifier_summary = st.container()
chatbot = st.container()

def process_attributes(df):
    
       
    # Rename the column names with under score
    df.rename(columns={'Data':'date', 'Countries':'country', 'Genre':'gender', 'Employee or Third Party':'employee type'}, inplace=True)
    df.rename(columns=lambda s: s.lower().replace(' ', '_'), inplace=True)
    
    # Lets divide Date column in Day, Month, Year, WeekDay, WeekOfYear and Quarter
    df['year']          = df['date'].apply(lambda x : x.year)
    df['month']         = df['date'].apply(lambda x : x.month)
    df['day']           = df['date'].apply(lambda x : x.day)
    df['weekday']       = df['date'].apply(lambda x : x.day_name())
    df['week_of_year']  = df['date'].apply(lambda x : x.weekofyear)
    df['quarter']       = df.date.dt.quarter
    
    
    
    # Label Accident Level to numeric category
    replace_accident_level = {'I': 1,'II': 2,'III': 3,'IV': 4,'V': 5}
    
    # Mapping Accident level into the dataframe
    df['accident_level'] = df['accident_level'].map(replace_accident_level)
    
    df_ind_acc["critical_risk"] = df_ind_acc["critical_risk"].apply( lambda x : 'Not applicable' if x == '\nNot applicable' else x)
    
    
    
        # Label Potential Accident Level to numeric category
    replace_potential_accident_level = {'I': 1,'II': 2,'III': 3,'IV': 4,'V': 5,'VI': 6}
    
    # Label Local to numeric category
    replace_local = {'Local_01': 1,'Local_02': 2,'Local_03': 3,'Local_04': 4,'Local_05': 5,'Local_06': 6,'Local_07': 7,'Local_08': 8,'Local_09': 9,'Local_10': 10,'Local_11': 11,'Local_12': 12}
    # Mapping Local into the dataframe
    df['local'] = df['local'].map(replace_local)
    
    df.drop_duplicates(inplace=True)
    # Delete temporary values because no more used
    del replace_local, replace_accident_level, replace_potential_accident_level

    return df

def preprocess_text(text):
  # lower case
  clean_text = text.lower()

  # remove dates
  date_regex = r"[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}"
  clean_text = re.sub( date_regex, "", clean_text)

  # remove time
  time_regex = r"(?i)([0-1]?[0-9]|[2][0-3]):([0-5]?[0-9])(.[am|pm]{2,2})?"
  clean_text = re.sub( time_regex, "", clean_text)

  # remove special characters
  specialchar_regex = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
  clean_text = re.sub( specialchar_regex, "", clean_text)

  # removing punctuations
  clean_text=re.sub(r'[?|!|\'|"|#|;~^*&$]',r'',clean_text)
  clean_text=re.sub(r'[.|,|)|(|\|/]',r'',clean_text)  

  # removing number - taking care of not removing equirement numbers starting with alphabets e.g AEQ-819, Nv.1850
  num_regex = r'[^a-zA-z.,!?/:;\"\'\s]'
  clean_text=re.sub(num_regex, r'',clean_text)
  
  
  # remove non-ascii words
  clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

  # remove all stopwords
  stop_words=nltk.corpus.stopwords.words('english')
  stop_words.extend(['cm', 'kg', 'tn', 'ton', 'diameter', 'meters', 'cx', 'nv', 'gallons', 'approx', 'm', 'mr', 'pound', 'h', 'hour', 'x', 'da', 'activity', 'area', 'time', 'work',
    'moment','causing','employee','support','approximately','circumstances','performing','performed','employees', 'use', 'company', 'made', 'two', 'one', 'using'
    'company','activities','moments','pm','due', 'came', 'moment', 'activity', 'report', 'described', 'medical', 'center', 'generating', 'immediately', 'event'])
  stop_words = set(stop_words)
  clean_text = ' '.join([words for words in clean_text.split() if words not in stop_words])

  #lemmatizing & stemming
  lemmatizer = stem.WordNetLemmatizer()
  lem = [lemmatizer.lemmatize(i) for i in tokenize.word_tokenize(clean_text) if i not in stop_words]
  
  clean_text = " ".join(lem)
  
  return clean_text


def display_metrics(yTest, yPred, printStats=True):
    """
    Print the metric score
    """
    accuracy = np.round(accuracy_score(yTest, yPred), 3)
    f1 = np.round(f1_score(yTest, yPred, average='weighted'), 3)
    precision = np.round(precision_score(yTest, yPred, average='weighted', zero_division=1), 3)
    recall = np.round(recall_score(yTest, yPred, average='weighted'), 3)
    #roc = np.round(roc_auc_score(yTest, y_pred_proba, multi_class="ovr", average='weighted'), 3)
    if(printStats):
      print('Accuracy score          : ', accuracy)
      print('F1 score                : ', f1, "(weighted)")
      print('Precision score         : ', precision, "(weighted)")
      print('Recall score            : ', recall, "(weighted)")
#print('ROC-AUC score (OVR)     : ', roc, "(weighted)")
    return accuracy, f1, precision, recall #, roc

def display_comparision(le, label, actual, predicted, num_rows=10):
    """
    Print comparision of top n-rows for actual vs prediccated
    """
    label = label[:num_rows]
    actual = actual[:num_rows]
    predicted = predicted[:num_rows]
    df_comparision = pd.DataFrame({'label': label,'Actual': le.inverse_transform(actual), 'Predicted': le.inverse_transform(predicted)})
    return df_comparision    



def ml_classifer(a):
    vectorizer = TfidfVectorizer(ngram_range=(2,2))
    
        # defining pipeline
    nlp_pipeline_multilabel = Pipeline([
        ('vectorizer', vectorizer),
        ('clf', a)
        ])

    nlp_pipeline_multilabel.fit(X_train, y_train)
    y_pred = nlp_pipeline_multilabel.predict(X_test)
    return display_metrics(y_test, y_pred)


def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

def f1_metric (precision, recall):
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

# defining threshold for prediction
def predictLabelForGivenThreshold(threshold, y_scores):
    y_pred=[]
    for sample in  y_scores:
        y_pred.append([1 if i>=threshold else 0 for i in sample ] )
    return np.array(y_pred)

# Performing cross validation for models
def mutlilable_cross_val(model, X_train, y_train, X_test, y_test, n_splits =3, n_epochs=10, threshold=.8, n_batch_size=64, callbacks=[], verbose =0):
  # threshold - for prediction if score is above 80% then consider 1 else 0
  mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
  i = 1
  accuracy_score = []
  f1_score = []
  precision_score =[]
  recall_score = []
  scores = {}
  
  history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = n_batch_size, validation_data = (X_test, y_test), callbacks = callbacks, verbose=verbose, use_multiprocessing=True)
  y_pred = model.predict(X_test)

  for train_index, test_index in mskf.split(X_train, y_train):
      print("[Info] evaluating batch", i)

      XX_train, XX_test = X_train[train_index], X_train[test_index]
      yy_train, yy_test = y_train[train_index], y_train[test_index]

      model.fit(XX_train, yy_train, epochs = n_epochs, batch_size = n_batch_size, validation_data = (XX_test, yy_test), callbacks = callbacks, verbose=verbose, use_multiprocessing=True)

      i +=1
      y_pred = model.predict(X_test_vec)
      y_pred = predictLabelForGivenThreshold(threshold, y_pred)
#y_pred = y_pred.round()

      acc_scr, f1_scr, prec_scr, recall_scr = display_metrics(y_test, y_pred, printStats=False)
      accuracy_score.append(acc_scr)
      f1_score.append(f1_scr)
      precision_score.append(prec_scr)
      recall_score.append(recall_scr)
  
  
  scores["accuracy"] = accuracy_score
  scores["F1"] = f1_score
  scores["precision"] = precision_score
  scores["recall"] = recall_score
  scores["predictions"] = y_pred
  scores["history"] = history
  return scores

 # model checkpoint and saving model
checkpoint_nn = ModelCheckpoint(
'nn_model.h5', 
monitor = 'val_loss', 
verbose = 0, 
save_best_only = True
)

checkpoint_lstm = ModelCheckpoint(
'lstm_model.h5', 
monitor = 'val_loss', 
verbose = 0, 
save_best_only = True
)

# defining early stopping
stop = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.001)

# reducing LR based on validation loss threshold
reduce_lr = ReduceLROnPlateau(
monitor = 'val_loss', 
factor = 0.2, 
verbose = 1, 
patience = 5,                        
min_lr = 0.001
)

# metrics to capture
METRICS = [
  keras.metrics.TruePositives(name='tp'),
  keras.metrics.FalsePositives(name='fp'),
  keras.metrics.TrueNegatives(name='tn'),
  keras.metrics.FalseNegatives(name='fn'), 
  keras.metrics.BinaryAccuracy(name='accuracy'),
  keras.metrics.Precision(name='precision'),
  keras.metrics.Recall(name='recall'),
  keras.metrics.AUC(name='auc', multi_label=True),
  keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
  
# create neural networks classifier
def glove_nn(embedding_matrix, n_inputs , n_outputs, metrics=METRICS, output_bias = None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)

  model = Sequential()
  model.add(Embedding(
        input_dim=embedding_matrix.shape[0], 
        output_dim=embedding_matrix.shape[1], 
        weights = [embedding_matrix], 
        input_length=n_inputs
    ))

  model.add(GlobalMaxPool1D())
  model.add(BatchNormalization())
  model.add(Dense(n_inputs, activation = "relu"))
  model.add(BatchNormalization())
  model.add(Dense(n_outputs, activation = 'sigmoid', bias_initializer=output_bias))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics = metrics)
  return model

def glove_lstm(embedding_matrix, n_inputs , n_outputs, metrics=METRICS, output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()

    model.add(Embedding(
    input_dim=embedding_matrix.shape[0], 
    output_dim=embedding_matrix.shape[1], 
    weights = [embedding_matrix], 
    input_length=n_inputs
))

    model.add(Bidirectional(LSTM(
    n_inputs, 
    return_sequences = True, 
    recurrent_dropout=0.2
)))

    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_inputs*2, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(n_inputs, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs, activation = 'sigmoid', bias_initializer=output_bias))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics =metrics)

    return model

def remove_additional_words(text):
    stop_words=nltk.corpus.stopwords.words('english')
    stop_words.extend(['cm', 'kg', 'tn', 'ton', 'diameter', 'meters', 'cx', 'nv', 'gallons', 'approx', 'm', 'mr', 'pound', 'h', 'hour', 'x', 'da', 'activity', 'area', 'time', 'work',
        'moment','causing','employee','support','approximately','circumstances','performing','performed','employees', 'use', 'company', 'made', 'two', 'one', 'using'
        'company','activities','moments','pm','due', 'came', 'moment', 'activity', 'report', 'described', 'medical', 'center', 'generating', 'immediately', 'event'])
    stop_words = set(stop_words)
    clean_text = ' '.join([words for words in text.split() if words not in stop_words])
    return clean_text




def clean_up_sentence(sentence):
    sentence = preprocess_text(sentence)
    sentence_words = remove_additional_words(sentence)
    return sentence_words


def predict_potential_accident_level(inputText, model, binarizer):
    input = []
    threshold=.8
    #clean up input text 
    data = pd.Series(clean_up_sentence(inputText))

    #converting to vector
    data_vec = pad_sequences(embed(data), length_long_sentence, padding='post')
    input.append(data_vec)
    # predict for given input
    y_pred = model.predict(input)
    #binarizer.inverse_transform(test_pred)
    potential_acc_level = predictLabelForGivenThreshold(threshold, y_pred)
    # preforming inverse transformation to get the label text
    pred_result = binarizer.inverse_transform(potential_acc_level)
    # Get potential accident level and critical risk labels
    pred_acc_lvl = ""
    pred_critical_risk = ""
    for val in pred_result[0]:
        if len(val) <3:
            pred_acc_lvl = np.array(val)
        elif len(val) >3:
            pred_critical_risk = np.array(val)
    
    if pred_acc_lvl != "":
        is_present = np.isin(pred_acc_lvl, potential_accident_level)
        pred_acc_lvl = np.array_str(pred_acc_lvl) if len(pred_acc_lvl[is_present]) else ''
        
    if pred_critical_risk != "":
        is_present = np.isin(pred_critical_risk, critical_risk)
        pred_critical_risk = np.array_str(pred_critical_risk) if len(pred_critical_risk[is_present]) else ''
    return pred_acc_lvl, pred_critical_risk

file_location="Data/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.xlsx"

with st.sidebar:
    selected = option_menu(menu_title = 'Main Menu',
                           options = ["About the Project","Data set introduction","Exploratory Data analysis","NLP Text Pre-Processing","Download Cleansed Data","Machine Learning Classifiers","Neural Network","LSTM Classifier","Classifier Summary & Pickling","Chat BOT Assistant"],
                           default_index = 0)

    if selected == "About the Project":
        
        with ProjectInformation:
            st.title("Industrial safety NLP based Chatbot")
            
            st.header("Domain")
            
            st.write("""
                <p style='text-align: justify;'>
                Industrial safety. NLP based Chatbot
             </p>
                """, unsafe_allow_html=True)
                
            st.header("Context")            
            st.write("""
                <p style='text-align: justify;'>
                The database comes from one of the biggest industries in Brazil and in the world. It is an urgent need for industries/companies around the globe to understand why employees still suffer some injuries/accidents in plants. Sometimes they also die in such an environment.
             </p>
                """, unsafe_allow_html=True)

            st.header("Data Description")            
            st.write("""
                <p style='text-align: justify;'>
                This database is basically records of accidents from 12 different plants in 03 different countries where every line in the data is an occurrence of an accident.
             </p>
                """, unsafe_allow_html=True)
                
            st.header("Columns description")            
            st.write("""
                <p style='text-align: justify;'>
                Data: timestamp or time/date information 
                
                Countries: which country the accident occurred (anonymised) 
                
                Local: the city where the manufacturing plant is located (anonymised) 
                
                Industry sector: which sector the plant belongs to 
                
                Accident level: from I to VI, it registers how severe was the accident (I means not severe but VI means very severe)
                
                Potential Accident Level: Depending on the Accident Level, the database also registers how severe the accident could have been (due to other factors involved in the accident)
                
                Genre: if the person is male of female 
                
                Employee or Third Party: if the injured person is an employee or a third party
                
                Critical Risk: some description of the risk involved in the accident 
                
                Description: Detailed description of how the accident happened. 
                
                Link to download the dataset:  https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database
             </p>
                """, unsafe_allow_html=True)
                
            st.header("Project Objective")
 
            st.write("""
                <p style='text-align: justify;'>
                Objective of current project is to analyse accident data, help manufacturing plants by providing insight into minimize/avoid accident(s) and save lives.

                Design a ML/DL based chatbot utility which can help the professionals to highlight the safety risk as per the incident description.

             </p>
                """, unsafe_allow_html=True)
                

    elif selected == "Data set introduction":    

        with data:
            st.title("Import Data for Analysis")
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                st.title("Displaying the loaded data")    
                df_ind_acc = pd.read_excel(uploaded_file)
                hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """
        
        # Inject CSS with Markdown
                #st.markdown(hide_table_row_index, unsafe_allow_html=True)
        
        # Display a static table
                st.write(df_ind_acc.head(5))
                st.write('Total Number of Rows:', df_ind_acc.shape[0])
                st.write('Total Number of Columns:', df_ind_acc.shape[1])
                

        
        ################Data Pre-Processing############################################        
                
                
                df_ind_acc.pop("Unnamed: 0")
                df_ind_acc = process_attributes(df_ind_acc) 
                
                st.title("After Data Modification and column renaming:")
                st.write(df_ind_acc.head())
                st.write('Total Number of Rows:', df_ind_acc.shape[0])
                st.write('Total Number of Columns:', df_ind_acc.shape[1])  
                
                st.header("Data Cleanup: Summary")
            
                st.write("""
                <p style='text-align: justify;'>
                ✅ Column conversion - Converted the Date column to datetime type
                
                ✅ Check for missing values - No missing values found
                
                ✅ Check for duplicate value - Found 7 duplicate rows, deleted duplicate rows
             </p>
                """, unsafe_allow_html=True)
        
        
        ################MText Pre- Processing############################################      
            

              
                df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
                st.title("After Text Pre-Processing:")
                st.write(df_ind_acc.head())
                st.write('Total Number of Rows:', df_ind_acc.shape[0])
                st.write('Total Number of Columns:', df_ind_acc.shape[1])
                
                st.header("Data Cleanup: Summary")
            
                st.write("""
                <p style='text-align: justify;'>
                ✅ Remove unused column - Unnamed: 0
                
                ✅ Rename column name - renamed column to make it uniformed and easy to refer
                
                ✅ Convert values to standard format - Converted category values of Local, Accident Level and Potential Accident Level
                
                ✅ Creating on new feature - New features created from date like, day, month, week, week of year, quarter and year
                
                ✅ Description NLP preprocessing like converting to lower case,lemmantization,removing stop words including domain specific,remove dates,time,special character etc                              
                    

             </p>
                """, unsafe_allow_html=True)
        
        
            else:
                st.warning("you need to upload a csv or excel file.")
                my_df  = pd.DataFrame()

################Univariate Analysis############################################    
    elif selected == "Exploratory Data analysis": # Function to plot pie chat for all the appropriate categorical features.
    
        with EDA:
            
           
            df_ind_acc = pd.read_excel(file_location)
            df_ind_acc.pop("Unnamed: 0")
            df_ind_acc = process_attributes(df_ind_acc)
            df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
            
    
            def plot_univariate(plot):
                df_plot = pd.DataFrame()
                if(isinstance(plot, pd.Series)):
                    df_plot = pd.DataFrame(plot)
    
                for col in df_plot.columns[0:]:
                    df = df_plot.copy(deep=True)
    
            #Pie plots need data to be arranged in terms of the pie sizes, hence use groupby to get sizes of each group
                    df.insert(0,'freq', 1) # Insert a column for the frequency of the group
                    df.insert(1,'%', 1) # Insert a column for the %size of the group
    
                    df = df[[col, 'freq', '%']] # The data frame consists of just the required columns
                    df = df.groupby(col).agg(sum) # Groupby each column by the groups, with values equal to sum of group 
    
                    values = df['freq'].values
                    labels = df.index.values
    
                    if (len(values) < 15):
                #uses Plot.ly go
                        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],)
                        fig.add_trace(go.Pie(labels=labels, values=values, hole=.3), row=1, col=1)
                        fig.add_trace(go.Histogram(x=df_plot[col], name="Total Count", texttemplate="%{y}", 
                                        textfont_size=20, marker=dict(color="#4CB391")), row=1, col=2)
                    else:
                        fig = go.Figure(data=[go.Bar(y=values, x=labels, orientation = "v")])
                        #fig.show()
                        #st.plotly_chart(fig, use_container_width=True)
                    fig.update_layout(title=go.layout.Title(text=col.title().replace("_", " "),x = 0, font=dict(size=50,color='red')), title_x=0.5)
                    fig.update_xaxes(showgrid=False)
                    #fig.show()
                    st.plotly_chart(fig, use_container_width=True)
                
           
            st.header("Univariate Analysis")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            option = st.selectbox(
         'Univariate Analysis?',
         ('Please Select a Value from Drop Down','country', 'industry_sector', 'local','gender','employee_type','critical_risk','potential_accident_level'))
            
            if option == "Please Select a Value from Drop Down":
                st.warning("Choose a column for Analysis")
                my_df  = pd.DataFrame()
            else:
                plot_univariate(df_ind_acc[option])
    
    
    ################MultiVariate Analysis############################################
                
            st.header("Multivariate Analysis")
            def plot_histogram(x, hue, barmode, title=""):
                title = x.name.title() if title == "" else title 
                fig = plt.figure(figsize=(10, 4))
                fig = px.histogram(x=x, color=hue, barmode=barmode, labels={'x':x.name}, height=400, text_auto=True)
                fig.update_layout(title=go.layout.Title(text= title,x = 0, font=dict(size=50,color='red')), title_x=0.5)
                fig.update_xaxes(automargin=True)
                #fig.show()
                st.plotly_chart(fig, use_container_width=True)
                
                
            option = st.selectbox(
         'Multivariate Analysis?',
         ('Please Select a Value from Drop Down','country&industry_sector', 'country&potential_accident_level', 
          'industry_sector&potential_accident_level','potential_accident_level&employee_type'))
            
            if option == "Please Select a Value from Drop Down":
                st.warning("Choose a column for Analysis")
                my_df  = pd.DataFrame()
            else:
                first = option.split('&')    
                plot_histogram(df_ind_acc[first[0]], df_ind_acc[first[1]], 'group',option)
    
    
    ################Count Plot############################################        
            
            st.header("Date Analysis")
            def countPlot(df,a,b,option):
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x =df[a],hue=df[b])
                fig.suptitle(option)
                st.pyplot(fig)
            
            option = st.selectbox(
         'Count Plot?',
         ('Please Select a Value from Drop Down','day&year', 'weekday&year'))
    
            
            if option == "Please Select a Value from Drop Down":
                st.warning("Choose a column for Analysis")
                my_df  = pd.DataFrame()
            else:
                first = option.split('&')
                a = first[0]
                b = first[1]
                countPlot(df_ind_acc,a,b,option)    
            
            
            
            
#####################################################################################################

            st.title("EDA Summary")
            
            st.write("""
                 ## Country
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -The most affected country from the above dataset is country 1 with around 59% of the accidents.
                    
                    -31% of the accidents occurred in country 2.
                    
                    -10% of the accidents occurred in country 3. 
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Industry Sector
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Mostly affected sector is Mining sector. 57% of accidents occur in Mining sector.
                    
                    -32% of Metals industry affected.
                    
                    -12% of other industry affected.
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Local
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Most accidents happened in Local_03 which is 21.18% .
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Potential Accident Level
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Most belongs to level IV which is equivalent to 34% of total potential accidents.
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Gender
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Most affected worker's in accidents are male which is equivalent to 94.8%.
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Employee Type
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Employee type of Third party is most prone to Accident risk.
                    
                    -Most affected Employee type are Third party workers which is equivalent to 45%.
                    
                    -42% of Employees affected.
                    
                    -13% of Third Party(Remote) affected.
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Critical Risk
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Number of incidents by each type of critical risk, Others tops the list and followed by Pressed
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Potential Accident Level + Date
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -We have partial data for 2017, upto 7th month
                    
                    -Accident trend shows, that smaller number of accidents in 2017
    
                    -Most accidents happened in year 2016 which is equivalent to 67% .
    
                    -Most accidents happened in Feb month which is equivalent to 14%.
    
                    -Most accidents happened in Thursday which is equivalent to 19%.
    
                    -Number of accidents is high in particular days like 4, 8 and 16 of the months.
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Country and Industry
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -Country 3 has only other industry
                    
                    -Country 1 and Country 2 has all three types of industries others being the least
                 </p>
                    """, unsafe_allow_html=True)
                    
            st.write("""
                 ## Potential Accident Level and Accident Level
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    -As Potential accident level increases Accident level also increases
                 </p>
                    """, unsafe_allow_html=True)

#########################################NLP Processing##########################################

    elif selected == "NLP Text Pre-Processing": # Function to plot pie chat for all the appropriate categorical features.
    
        with NLP:
            
            df_ind_acc = pd.read_excel(file_location)
            df_ind_acc.pop("Unnamed: 0")
            df_ind_acc = process_attributes(df_ind_acc)
            df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
            
            col1, col2 = st.columns(2)
            
            with col1:          
                st.header("Words in description")
                image = Image.open('images/Words_Description.png')
                st.image(image, use_column_width=True)
                
            with col2:          
                st.header("Average Word Length")
                image = Image.open('images/WordLength.png')
                st.image(image, use_column_width=True)
                
                    
            st.header("Top ngrams")
            image = Image.open('images/bigrams.png')
            st.image(image, use_column_width=True)

                
            st.header("Word Cloud")
            wordcloud = WordCloud(width = 1500, height = 800, random_state=0, background_color='black', colormap='rainbow', \
                          min_font_size=5, max_words=300, collocations=False, stopwords = STOPWORDS).generate(" ".join(df_ind_acc['description_processed'].values))
            plt.figure(figsize=(15,10))
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot()
            
            
            st.header("Style Cloud")
            # Setting the image - 
            image = Image.open('images/user-safety.png')
    
            # Setting the image width -
            st.image(image, use_column_width=False)
    
    
            st.title("Text Pre-Processing Summary")
            
            st.write("""
                 ## Summary
                 """)
            st.write("""
                    <p style='text-align: justify;'>
                    Based on incident description we could identify accident keyword contains:
    
                        -Employee Role - Operator, collaborator, Worker, Assistant, Mechanic
                        
                        -Body parts - left, right, Hand, Leg, finger, face, Foot
                        
                        -Equipment Type - Truck, Pipe, Pump, Drill, Car, tube
                        
                        -Accident involved - Hit, Fall, cutting
                 </p>
                    """, unsafe_allow_html=True)   

    elif selected == "Download Cleansed Data": # Function to plot pie chat for all the appropriate categorical features.
    
        with download:
            
            df_ind_acc = pd.read_excel(file_location)
            df_ind_acc.pop("Unnamed: 0")
            df_ind_acc = process_attributes(df_ind_acc)
            df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
    
            st.header("File Download Post Data Cleansing")
            #@st.cache
            def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
             return df.to_csv().encode('utf-8')
    
            csv = convert_df(df_ind_acc)
    
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='Cleansed_File.csv',
                mime='text/csv',
                )

    elif selected == "Machine Learning Classifiers": # Function to plot pie chat for all the appropriate categorical features.
    
        

        with machinelearning_model_training:
                    st.title("Machine Learning Classifiers")   
            
                    genre = st.radio(
             "Do you want to use the Existing Cleansed or upload a new data for analysis",
             ('Do not want to run','Yes', 'No'))
        
                    if genre == 'Yes':
                        df_ind_acc = pd.read_excel(file_location)
                        df_ind_acc.pop("Unnamed: 0")
                        df_ind_acc = process_attributes(df_ind_acc)
                        df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
                        st.title("Existing Cleansed data:")
                        st.write(df_ind_acc.head())
                        st.write('Total Number of Rows:', df_ind_acc.shape[0])
                        st.write('Total Number of Columns:', df_ind_acc.shape[1])
                        
                                        # Upsampling strategry using resmapling the dataset
                        df_minority_sampled = pd.DataFrame()
                        df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
                
                        UPSAMPLEPCT = .1
                        SEED = 45 
                
                        for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                            GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                            resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                            df_minority_sampled = df_minority_sampled.append(resampled)
                
                        df_upsampled = pd.concat([df_majority, df_minority_sampled])
                
                        # Shuffle all the samples
                        df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                        
                       
                        X= df_upsampled.description_processed
                        y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                        
                        label_counts=dict()
                
                        for labels in y.values:
                            for label in labels:
                                if label in label_counts:
                                    label_counts[str(label)]+=1
                                else:
                                    label_counts[str(label)]=1
                                    
                        # Transform between iterable of iterables and a multilabel format
                        binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
                
                        y_train = binarizer.fit_transform(y_train)
                        y_test = binarizer.transform(y_test)
                        
                        values = ['Please Select a Value from Drop Down','Random Forest Classifier', 'Logistic Regression', 
                      'BinaryRelevance-GaussianNB','Chain-Logistic Regression']                   
                        option = st.selectbox('Choose your Machine Learning Classifier?', values,index=0 )
                        
                  
                        
                        if option == "Random Forest Classifier":
                            a = ml_classifer(OneVsRestClassifier(RandomForestClassifier()))
                            st.subheader('Accuracy Score of the Model is:')
                            st.write(a[0])
                            st.subheader('F1 Score of the Model is:')
                            st.write(a[1])
                            st.subheader('Precision Score of the Model is:')
                            st.write(a[2])
                            st.subheader('Recall Score of the Model is:')
                            st.write(a[3])           
                        
                        
                        elif option=="Logistic Regression":                   
                            a = ml_classifer(OneVsRestClassifier(LogisticRegression(class_weight='balanced')))
                            st.subheader('Accuracy Score of the Model is:')
                            st.write(a[0])
                            st.subheader('F1 Score of the Model is:')
                            st.write(a[1])
                            st.subheader('Precision Score of the Model is:')
                            st.write(a[2])
                            st.subheader('Recall Score of the Model is:')
                            st.write(a[3])           
                        
                        
                        elif option == "BinaryRelevance-GaussianNB":                   
                            a = ml_classifer(BinaryRelevance(GaussianNB()))
                            st.subheader('Accuracy Score of the Model is:')
                            st.write(a[0])
                            st.subheader('F1 Score of the Model is:')
                            st.write(a[1])
                            st.subheader('Precision Score of the Model is:')
                            st.write(a[2])
                            st.subheader('Recall Score of the Model is:')
                            st.write(a[3])           
                        
                            
                        elif option=="Chain-Logistic Regression":                   
                            a = ml_classifer(ClassifierChain(LogisticRegression(class_weight='balanced')))
                            st.subheader('Accuracy Score of the Model is:')
                            st.write(a[0])
                            st.subheader('F1 Score of the Model is:')
                            st.write(a[1])
                            st.subheader('Precision Score of the Model is:')
                            st.write(a[2])
                            st.subheader('Recall Score of the Model is:')
                            st.write(a[3])           
                        
                        
                        else:
                            st.warning("Choose your Machine Learning Classifier")
                        
                    elif genre == 'No':
        
                        st.title("Upload Cleansed Data")
                        spectra = st.file_uploader("upload file", type={"csv", "txt"})
                        
                        if spectra is not None:
                            df_ind_acc = pd.read_csv(spectra)
                            st.write(df_ind_acc.head())
                            st.write('Total Number of Rows:', df_ind_acc.shape[0])
                            st.write('Total Number of Columns:', df_ind_acc.shape[1])
                    
                            df_minority_sampled = pd.DataFrame()
                            df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
                    
                            UPSAMPLEPCT = .1
                            SEED = 45 
                    
                            for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                                GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                                resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                                df_minority_sampled = df_minority_sampled.append(resampled)
                    
                            df_upsampled = pd.concat([df_majority, df_minority_sampled])
                    
                            # Shuffle all the samples
                            df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                            
                           
                            X= df_upsampled.description_processed
                            y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                            
                            label_counts=dict()
                    
                            for labels in y.values:
                                for label in labels:
                                    if label in label_counts:
                                        label_counts[str(label)]+=1
                                    else:
                                        label_counts[str(label)]=1
                                        
                            # Transform between iterable of iterables and a multilabel format
                            binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
                    
                            y_train = binarizer.fit_transform(y_train)
                            y_test = binarizer.transform(y_test)
                            
                            values = ['Please Select a Value from Drop Down','Random Forest Classifier', 'Logistic Regression', 
                          'BinaryRelevance-GaussianNB','Chain-Logistic Regression']                   
                            option = st.selectbox('Choose your Machine Learning Classifier?', values,index=0 )
                            
                      
                            
                            if option == "Random Forest Classifier":
                                a = ml_classifer(OneVsRestClassifier(RandomForestClassifier()))
                                st.subheader('Accuracy Score of the Model is:')
                                st.write(a[0])
                                st.subheader('F1 Score of the Model is:')
                                st.write(a[1])
                                st.subheader('Precision Score of the Model is:')
                                st.write(a[2])
                                st.subheader('Recall Score of the Model is:')
                                st.write(a[3])           
                            
                            
                            elif option=="Logistic Regression":                   
                                a = ml_classifer(OneVsRestClassifier(LogisticRegression(class_weight='balanced')))
                                st.subheader('Accuracy Score of the Model is:')
                                st.write(a[0])
                                st.subheader('F1 Score of the Model is:')
                                st.write(a[1])
                                st.subheader('Precision Score of the Model is:')
                                st.write(a[2])
                                st.subheader('Recall Score of the Model is:')
                                st.write(a[3])           
                            
                            
                            elif option == "BinaryRelevance-GaussianNB":                   
                                a = ml_classifer(BinaryRelevance(GaussianNB()))
                                st.subheader('Accuracy Score of the Model is:')
                                st.write(a[0])
                                st.subheader('F1 Score of the Model is:')
                                st.write(a[1])
                                st.subheader('Precision Score of the Model is:')
                                st.write(a[2])
                                st.subheader('Recall Score of the Model is:')
                                st.write(a[3])           
                            
                                
                            elif option=="Chain-Logistic Regression":                   
                                a = ml_classifer(ClassifierChain(LogisticRegression(class_weight='balanced')))
                                st.subheader('Accuracy Score of the Model is:')
                                st.write(a[0])
                                st.subheader('F1 Score of the Model is:')
                                st.write(a[1])
                                st.subheader('Precision Score of the Model is:')
                                st.write(a[2])
                                st.subheader('Recall Score of the Model is:')
                                st.write(a[3])           
                            
                            
                            else:
                                st.warning("Choose your Machine Learning Classifier")
                                    
                                            # Upsampling strategry using resmapling the dataset
                                                
                        else:
                            st.warning("you need to choose a value to run ML/NN/NLP Classifier")
                            my_df  = pd.DataFrame()
                

                    
            
                    
                


    elif selected == "Neural Network":   
        with NN_model_training:
            st.title("Neural Network")
            
            genre1 = st.radio(
             "Do you want to use the Existing Cleansed or upload a new data for analysis of NN/LSTM Classifier",
             ('Do not want to run NN Model','Yes', 'No'))
            
                
            if genre1 == 'Yes':      
                
                df_ind_acc = pd.read_excel(file_location)
                df_ind_acc.pop("Unnamed: 0")
                df_ind_acc = process_attributes(df_ind_acc)
                df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
                
              
                df_minority_sampled = pd.DataFrame()
                df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
        
                UPSAMPLEPCT = .1
                SEED = 45 
        
                for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                    GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                    resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                    df_minority_sampled = df_minority_sampled.append(resampled)
        
                df_upsampled = pd.concat([df_majority, df_minority_sampled])
        
        # Shuffle all the samples
                df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                
               
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                
                label_counts=dict()
        
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
                            
                # Transform between iterable of iterables and a multilabel format
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
        
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
                
            
                word_tokenizer = Tokenizer()
                word_tokenizer.fit_on_texts(X)
                num_tokens = len(word_tokenizer.word_index) + 1
                
                longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
                length_long_sentence = len(word_tokenize(longest_train))
            
        
                
                # Using spacy as text vectorizer and generating embeddings
                spacy_nlp = spacy.load("en_core_web_md")
                embeddings_dictionary = dict()
            
                embedding_dim = len(spacy_nlp('The').vector)
                embedding_matrix = np.zeros((num_tokens, embedding_dim))
                for word, index in tqdm(word_tokenizer.word_index.items()):
                    embedding_matrix[index] = spacy_nlp(str(word)).vector
            
                #embedding_matrix.shape[0], num_tokens
                
                # Getting distinct labels
                label_counts=dict()
            
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
            
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
                
                # using MultiLabelBinarizer for encoding 
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
            
                X_train_vec = pad_sequences( embed(X_train), length_long_sentence, padding='post')
                X_test_vec = pad_sequences( embed(X_test), length_long_sentence, padding='post')
            
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
            
                n_outputs = y_train.shape[1]
                
                # calculating F1 metic from precision and recall
        
            
                   
        
            
                model_nn = glove_nn(embedding_matrix, length_long_sentence, y_train.shape[1])
                #st.write(model_nn.summary())
                
                n_epochs=50
                n_splits = 3
                scores_nn = mutlilable_cross_val(model_nn, X_train_vec, y_train, X_test_vec, y_test, n_epochs=n_epochs, callbacks= [reduce_lr, checkpoint_nn, stop], verbose=1)
                
                BATCH_SIZE = 1024
                baseline_results = model_nn.evaluate(X_test_vec, y_test, batch_size=BATCH_SIZE, verbose=0)
                model_metrics_nn = dict()
                for name, value in zip(model_nn.metrics_names, baseline_results):
                    model_metrics_nn[name] = value
                model_metrics_nn["F1"] = f1_metric(model_metrics_nn["precision"], model_metrics_nn["recall"])
                
                
                st.subheader('Neural Network Summary:')
                # Setting the image - 
                image = Image.open('images/simple_cnn.png')
            
                # Setting the image width -
                st.image(image, use_column_width=False)
                
              
                
                st.subheader('Accuracy Score of the Model is:')
                st.write(model_metrics_nn["accuracy"]*100)
                st.subheader('F1 Score of the Model is:')
                st.write(model_metrics_nn["F1"]*100)
                st.subheader('Precision Score of the Model is:')
                st.write(model_metrics_nn["precision"]*100)
                st.subheader('Recall Score of the Model is:')
                st.write(model_metrics_nn["recall"]*100)
                
        
            elif genre1 == 'No':
                
                st.title("Upload Cleansed Data")
                spectra = st.file_uploader("upload file", type={"csv", "txt"})
                if spectra is not None:
                    df_ind_acc = pd.read_csv(spectra)
                    st.write(df_ind_acc.head())
                    st.write('Total Number of Rows:', df_ind_acc.shape[0])
                    st.write('Total Number of Columns:', df_ind_acc.shape[1])
                    
                    df_minority_sampled = pd.DataFrame()
                    df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
            
                    UPSAMPLEPCT = .1
                    SEED = 45 
            
                    for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                        GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                        resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                        df_minority_sampled = df_minority_sampled.append(resampled)
            
                    df_upsampled = pd.concat([df_majority, df_minority_sampled])
            
            # Shuffle all the samples
                    df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                    
                   
                    X= df_upsampled.description_processed
                    y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                    
                    label_counts=dict()
            
                    for labels in y.values:
                        for label in labels:
                            if label in label_counts:
                                label_counts[str(label)]+=1
                            else:
                                label_counts[str(label)]=1
                                
                    # Transform between iterable of iterables and a multilabel format
                    binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
            
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)
                    
               
                    word_tokenizer = Tokenizer()
                    word_tokenizer.fit_on_texts(X)
                    num_tokens = len(word_tokenizer.word_index) + 1
                    
                    longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
                    length_long_sentence = len(word_tokenize(longest_train))
                
                   
                    # Using spacy as text vectorizer and generating embeddings
                    spacy_nlp = spacy.load("en_core_web_md")
                    embeddings_dictionary = dict()
                
                    embedding_dim = len(spacy_nlp('The').vector)
                    embedding_matrix = np.zeros((num_tokens, embedding_dim))
                    for word, index in tqdm(word_tokenizer.word_index.items()):
                        embedding_matrix[index] = spacy_nlp(str(word)).vector
                
                    #embedding_matrix.shape[0], num_tokens
                    
                    # Getting distinct labels
                    label_counts=dict()
                
                    for labels in y.values:
                        for label in labels:
                            if label in label_counts:
                                label_counts[str(label)]+=1
                            else:
                                label_counts[str(label)]=1
                
                    X= df_upsampled.description_processed
                    y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                    
                    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
                    
                    # using MultiLabelBinarizer for encoding 
                    binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
                
                    X_train_vec = pad_sequences( embed(X_train), length_long_sentence, padding='post')
                    X_test_vec = pad_sequences( embed(X_test), length_long_sentence, padding='post')
                
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)
                
                    n_outputs = y_train.shape[1]
                    
                    # calculating F1 metic from precision and recall
              
          
            
                
                    model_nn = glove_nn(embedding_matrix, length_long_sentence, y_train.shape[1])
                    #st.write(model_nn.summary())
                    
                    n_epochs=50
                    n_splits = 3
                    scores_nn = mutlilable_cross_val(model_nn, X_train_vec, y_train, X_test_vec, y_test, n_epochs=n_epochs, callbacks= [reduce_lr, checkpoint_nn, stop], verbose=1)
                    
                    BATCH_SIZE = 1024
                    baseline_results = model_nn.evaluate(X_test_vec, y_test, batch_size=BATCH_SIZE, verbose=0)
                    model_metrics_nn = dict()
                    for name, value in zip(model_nn.metrics_names, baseline_results):
                        model_metrics_nn[name] = value
                    model_metrics_nn["F1"] = f1_metric(model_metrics_nn["precision"], model_metrics_nn["recall"])
                    
                    
                    st.subheader('Neural Network Summary:')
                    # Setting the image - 
                    image = Image.open('images/simple_cnn.png')
                
                    # Setting the image width -
                    st.image(image, use_column_width=False)
                   
                    st.subheader('Accuracy Score of the Model is:')
                    st.write(model_metrics_nn["accuracy"]*100)
                    st.subheader('F1 Score of the Model is:')
                    st.write(model_metrics_nn["F1"]*100)
                    st.subheader('Precision Score of the Model is:')
                    st.write(model_metrics_nn["precision"]*100)
                    st.subheader('Recall Score of the Model is:')
                    st.write(model_metrics_nn["recall"]*100)
                    
        
            else:
                st.warning("you need to choose a value to run NN Classifier")
                my_df  = pd.DataFrame()
                
    elif selected == "LSTM Classifier":   
        with LSTM_model_training:
            st.title("LSTM Classifier")
            
            genre1 = st.radio(
             "Do you want to use the Existing Cleansed or upload a new data for analysis of NN/LSTM Classifier",
             ('Do not want to run LSTM Model','Yes', 'No'))
            
                
            if genre1 == 'Yes':      
                
                df_ind_acc = pd.read_excel(file_location)
                df_ind_acc.pop("Unnamed: 0")
                df_ind_acc = process_attributes(df_ind_acc)
                df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
                
                df_minority_sampled = pd.DataFrame()
                df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
        
                UPSAMPLEPCT = .1
                SEED = 45 
        
                for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                    GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                    resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                    df_minority_sampled = df_minority_sampled.append(resampled)
        
                df_upsampled = pd.concat([df_majority, df_minority_sampled])
        
        # Shuffle all the samples
                df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                
               
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                
                label_counts=dict()
        
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
                            
                # Transform between iterable of iterables and a multilabel format
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
        
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
                
           
                word_tokenizer = Tokenizer()
                word_tokenizer.fit_on_texts(X)
                num_tokens = len(word_tokenizer.word_index) + 1
                
                longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
                length_long_sentence = len(word_tokenize(longest_train))
            
               
                # Using spacy as text vectorizer and generating embeddings
                spacy_nlp = spacy.load("en_core_web_md")
                embeddings_dictionary = dict()
            
                embedding_dim = len(spacy_nlp('The').vector)
                embedding_matrix = np.zeros((num_tokens, embedding_dim))
                for word, index in tqdm(word_tokenizer.word_index.items()):
                    embedding_matrix[index] = spacy_nlp(str(word)).vector
            
                #embedding_matrix.shape[0], num_tokens
                
                # Getting distinct labels
                label_counts=dict()
            
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
            
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
                
                # using MultiLabelBinarizer for encoding 
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
            
                X_train_vec = pad_sequences( embed(X_train), length_long_sentence, padding='post')
                X_test_vec = pad_sequences( embed(X_test), length_long_sentence, padding='post')
            
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
            
                n_outputs = y_train.shape[1]
              
                st.title("LSTM")
                
                # create LSTM classifier

            
                model_lstm = glove_lstm(embedding_matrix, length_long_sentence, y_train.shape[1])
                model_lstm.summary()
                
                st.subheader('Bidirectional LSTM:')
                # Setting the image - 
                image = Image.open('images/Bidirectional_LSTM.png')
            
                # Setting the image width -
                st.image(image, use_column_width=False)
                
                n_epochs=50
                n_splits = 3
                scores_lstm = mutlilable_cross_val(model_lstm, X_train_vec, y_train, X_test_vec, y_test, n_epochs=n_epochs, callbacks= [reduce_lr, checkpoint_lstm, stop], verbose=1)
                
                BATCH_SIZE = 1024
                baseline_results = model_lstm.evaluate(X_test_vec, y_test, batch_size=BATCH_SIZE, verbose=0)
                model_metrics_lstm = dict()
                for name, value in zip(model_lstm.metrics_names, baseline_results):
                    model_metrics_lstm[name] = value
                model_metrics_lstm["F1"] = f1_metric(model_metrics_lstm["precision"], model_metrics_lstm["recall"])
                
                st.subheader('Accuracy Score of the Model is:')
                st.write(model_metrics_lstm["accuracy"]*100)
                st.subheader('F1 Score of the Model is:')
                st.write(model_metrics_lstm["F1"]*100)
                st.subheader('Precision Score of the Model is:')
                st.write(model_metrics_lstm["precision"]*100)
                st.subheader('Recall Score of the Model is:')
                st.write(model_metrics_lstm["recall"]*100)    
                
                
            elif genre1 == 'No':
                
                st.title("Upload Cleansed Data")
                spectra = st.file_uploader("upload file", type={"csv", "txt"})
                if spectra is not None:
                    df_ind_acc = pd.read_csv(spectra)
                    st.write(df_ind_acc.head())
                    st.write('Total Number of Rows:', df_ind_acc.shape[0])
                    st.write('Total Number of Columns:', df_ind_acc.shape[1])
                    
                    df_minority_sampled = pd.DataFrame()
                    df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
            
                    UPSAMPLEPCT = .1
                    SEED = 45 
            
                    for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                        GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                        resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                        df_minority_sampled = df_minority_sampled.append(resampled)
            
                    df_upsampled = pd.concat([df_majority, df_minority_sampled])
            
            # Shuffle all the samples
                    df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                    
                   
                    X= df_upsampled.description_processed
                    y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                    
                    label_counts=dict()
            
                    for labels in y.values:
                        for label in labels:
                            if label in label_counts:
                                label_counts[str(label)]+=1
                            else:
                                label_counts[str(label)]=1
                                
                    # Transform between iterable of iterables and a multilabel format
                    binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
            
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)
                    
               
                    word_tokenizer = Tokenizer()
                    word_tokenizer.fit_on_texts(X)
                    num_tokens = len(word_tokenizer.word_index) + 1
                    
                    longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
                    length_long_sentence = len(word_tokenize(longest_train))
                
                   
                    # Using spacy as text vectorizer and generating embeddings
                    spacy_nlp = spacy.load("en_core_web_md")
                    embeddings_dictionary = dict()
                
                    embedding_dim = len(spacy_nlp('The').vector)
                    embedding_matrix = np.zeros((num_tokens, embedding_dim))
                    for word, index in tqdm(word_tokenizer.word_index.items()):
                        embedding_matrix[index] = spacy_nlp(str(word)).vector
                
                    #embedding_matrix.shape[0], num_tokens
                    
                    # Getting distinct labels
                    label_counts=dict()
                
                    for labels in y.values:
                        for label in labels:
                            if label in label_counts:
                                label_counts[str(label)]+=1
                            else:
                                label_counts[str(label)]=1
                
                    X= df_upsampled.description_processed
                    y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                    
                    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
                    
                    # using MultiLabelBinarizer for encoding 
                    binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
                
                    X_train_vec = pad_sequences( embed(X_train), length_long_sentence, padding='post')
                    X_test_vec = pad_sequences( embed(X_test), length_long_sentence, padding='post')
                
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)
                
                    n_outputs = y_train.shape[1]
                  
                    st.title("LSTM")
                
                    model_lstm = glove_lstm(embedding_matrix, length_long_sentence, y_train.shape[1])
                    model_lstm.summary()
                    
                    st.subheader('Bidirectional LSTM:')
                    # Setting the image - 
                    image = Image.open('images/Bidirectional_LSTM.png')
                
                    # Setting the image width -
                    st.image(image, use_column_width=False)
                    
                    n_epochs=50
                    n_splits = 3
                    scores_lstm = mutlilable_cross_val(model_lstm, X_train_vec, y_train, X_test_vec, y_test, n_epochs=n_epochs, callbacks= [reduce_lr, checkpoint_lstm, stop], verbose=1)
                    
                    BATCH_SIZE = 1024
                    baseline_results = model_lstm.evaluate(X_test_vec, y_test, batch_size=BATCH_SIZE, verbose=0)
                    model_metrics_lstm = dict()
                    for name, value in zip(model_lstm.metrics_names, baseline_results):
                        model_metrics_lstm[name] = value
                    model_metrics_lstm["F1"] = f1_metric(model_metrics_lstm["precision"], model_metrics_lstm["recall"])
                    
                    st.subheader('Accuracy Score of the Model is:')
                    st.write(model_metrics_lstm["accuracy"]*100)
                    st.subheader('F1 Score of the Model is:')
                    st.write(model_metrics_lstm["F1"]*100)
                    st.subheader('Precision Score of the Model is:')
                    st.write(model_metrics_lstm["precision"]*100)
                    st.subheader('Recall Score of the Model is:')
                    st.write(model_metrics_lstm["recall"]*100)    
                    
                    

        
            else:
                st.warning("you need to choose a value to run LSTM Classifier")
                my_df  = pd.DataFrame()
                
    elif selected == "Classifier Summary & Pickling": 
        with classifier_summary:
            
            st.header("Summary of all Classifiers")
            image = Image.open('images/Summary.png')
            st.image(image, use_column_width=False)
            st.title("Based on above metric we are choosing LSTM classifier for as a final model and pickle it")
                     

            
            if st.button('Do you want to pickle the final model: Then click me'):
                df_ind_acc = pd.read_excel(file_location)
                df_ind_acc.pop("Unnamed: 0")
                df_ind_acc = process_attributes(df_ind_acc)
                df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
                
                df_minority_sampled = pd.DataFrame()
                df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
        
                UPSAMPLEPCT = .1
                SEED = 45 
        
                for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                    GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                    resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                    df_minority_sampled = df_minority_sampled.append(resampled)
        
                df_upsampled = pd.concat([df_majority, df_minority_sampled])
        
        # Shuffle all the samples
                df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
                
               
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
                
                label_counts=dict()
        
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
                            
                # Transform between iterable of iterables and a multilabel format
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
        
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
                
           
                word_tokenizer = Tokenizer()
                word_tokenizer.fit_on_texts(X)
                num_tokens = len(word_tokenizer.word_index) + 1
                
                longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
                length_long_sentence = len(word_tokenize(longest_train))
            
               
                # Using spacy as text vectorizer and generating embeddings
                spacy_nlp = spacy.load("en_core_web_md")
                embeddings_dictionary = dict()
            
                embedding_dim = len(spacy_nlp('The').vector)
                embedding_matrix = np.zeros((num_tokens, embedding_dim))
                for word, index in tqdm(word_tokenizer.word_index.items()):
                    embedding_matrix[index] = spacy_nlp(str(word)).vector
            
                #embedding_matrix.shape[0], num_tokens
                
                # Getting distinct labels
                label_counts=dict()
            
                for labels in y.values:
                    for label in labels:
                        if label in label_counts:
                            label_counts[str(label)]+=1
                        else:
                            label_counts[str(label)]=1
            
                X= df_upsampled.description_processed
                y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
                
                X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
                
                # using MultiLabelBinarizer for encoding 
                binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
            
                X_train_vec = pad_sequences( embed(X_train), length_long_sentence, padding='post')
                X_test_vec = pad_sequences( embed(X_test), length_long_sentence, padding='post')
            
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)
            
                n_outputs = y_train.shape[1]
              
               # create LSTM classifier
    
            
                model_lstm = glove_lstm(embedding_matrix, length_long_sentence, y_train.shape[1])
                model_lstm.summary()
                
               
                n_epochs=50
                n_splits = 3
                scores_lstm = mutlilable_cross_val(model_lstm, X_train_vec, y_train, X_test_vec, y_test, n_epochs=n_epochs, callbacks= [reduce_lr, checkpoint_lstm, stop], verbose=1)
                
                BATCH_SIZE = 1024
                baseline_results = model_lstm.evaluate(X_test_vec, y_test, batch_size=BATCH_SIZE, verbose=0)
                
                                    #st.header("Save the Pickle File")
                        # Save the weights
                model_lstm.save_weights('lstm_model.h5')
                model_lstm.save('lstm_model.h5')
                st.write("Pickle file has been saved in the name lstm_model.h5 to the source directory")

            
        
                    
    elif selected == "Chat BOT Assistant": 
        with chatbot:

            df_ind_acc = pd.read_excel(file_location)
            df_ind_acc.pop("Unnamed: 0")
            df_ind_acc = process_attributes(df_ind_acc)
            df_ind_acc['description_processed'] = df_ind_acc["description"].apply(preprocess_text)
    
    
    # load the saved model file
            new_model = tf.keras.models.load_model('lstm_model - Copy.h5')
            
            df_minority_sampled = pd.DataFrame()
            df_majority =  df_ind_acc[df_ind_acc["critical_risk"] =='Others']
        
            UPSAMPLEPCT = .1
            SEED = 45 
        
            for risk in df_ind_acc[df_ind_acc["critical_risk"] !='Others']["critical_risk"].unique():
                GrpDF = df_ind_acc[df_ind_acc["critical_risk"] == risk]
                resampled = resample(GrpDF, replace=True, n_samples=int(UPSAMPLEPCT * df_majority.shape[0]/(1-UPSAMPLEPCT)), random_state=SEED)
                df_minority_sampled = df_minority_sampled.append(resampled)
        
            df_upsampled = pd.concat([df_majority, df_minority_sampled])
        
        # Shuffle all the samples
            df_upsampled = resample(df_upsampled, replace=False, random_state=SEED)
            
           
            X= df_upsampled.description_processed
            y= df_upsampled.apply(lambda col : [col["critical_risk"], col["potential_accident_level"]],axis =1)
            
            potential_accident_level = df_upsampled.potential_accident_level.unique()
            critical_risk = df_upsampled.critical_risk.unique()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=90)
            
            label_counts=dict()
        
            for labels in y.values:
                for label in labels:
                    if label in label_counts:
                        label_counts[str(label)]+=1
                    else:
                        label_counts[str(label)]=1
                        
            # Transform between iterable of iterables and a multilabel format
            binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
        
            y_train = binarizer.fit_transform(y_train)
            y_test = binarizer.transform(y_test)
        #pickled_binarizer = pickle.load(open('binarizer.pkl', 'rb'))
        
                     
            word_tokenizer = Tokenizer()
            word_tokenizer.fit_on_texts(X)
            num_tokens = len(word_tokenizer.word_index) + 1
            
            longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
            length_long_sentence = len(word_tokenize(longest_train))
        
            
            # Bot response based on user inputs, predict the potential accident level and critical risk
            def chatbot_response(input_txt):
              pred_acc_lvl, pred_critical_risk = predict_potential_accident_level(input_txt, new_model, binarizer)
              repsonse = "Based on your inputs seems like incident belong to "
            
              if pred_acc_lvl != "":
                repsonse += "accident level '" + pred_acc_lvl  + "'"            
               
              rsp_len = len(repsonse)
              
              if pred_critical_risk != "":
                repsonse += " and " if rsp_len > 51 else "" 
                repsonse += "critical risk category '" + pred_critical_risk + "'"
              return repsonse
             
            greeting = ["hi", 
                      "how are you", 
                      "is anyone there", 
                      "hello", 
                      "whats up",
                      "hey",
                      "yo",
                      "listen", 
                      "please help me",
                      "i am learner from",
                      "i belong to",
                      "aiml batch",
                      "aifl batch",
                      "i am from",
                      "my pm is",
                      "blended",
                      "online",
                      "i am from",
                      "hey ya",
                      "talking to you for first time"]
            
            exit_message = ["thank you", 
                      "thanks", 
                      "cya",
                      "see you",
                      "later", 
                      "see you later", 
                      "goodbye", 
                      "i am leaving", 
                      "have a Good day",
                      "you helped me",
                      "thanks a lot",
                      "thanks a ton",
                      "you are the best",
                      "great help",
                      "too good",
                      "bye",
                      "you are a good learning buddy"]
            
            
            discussion = ["problem",
                      "i have a problem to discuss",
                      "i want to discuss a problem",
                      "tell me about potential accident level",
                      "want to tell you the problem",
                      "can we start to discuss the problem",
                      "discuss",
                      "help",
                      "need your help",
                      "need",
                      "issue",
                      "SOS",
                      "help",
                      "incident",
                      "accident"
                      ]
            
            bot_info = ["what is your name",
                      "who are you",
                      "name please",
                      "when are your hours of opertions", 
                      "what are your working hours", 
                      "hours of operation",
                      "working hours",
                      "hours"]
            
            end_convo = ["what the hell",
                      "bloody stupid bot",
                      "do you think you are very smart",
                      "screw you", 
                      "i hate you", 
                      "you are stupid",
                      "jerk",
                      "you are a joke",
                      "useless piece of shit","idiot"]
            
           
            
            message("PAGGS-BOT!!!! Your Personal Assistant. If you want to exit, type end \n\n") 
             # align's the message to the right
            
            if 'generated' not in st.session_state:
                st.session_state['generated'] = []

            if 'past' not in st.session_state:
                st.session_state['past'] = []
            
            user_input = st.text_input("Enter Your Text here")
            
            user_input = user_input.lower()
            
                                 
            message(user_input,is_user=True)
            
            if len(user_input)>1:
            
                if user_input == "end":
                    message("Thanks for reaching out to us, we wish to serve you better in the future", is_user=False)
                    
                elif user_input in greeting:
                    message("Hello! how can i help you ?", is_user=False)
                    
                elif user_input in exit_message:
                    message("I hope I was able to assist you, Good Bye?", is_user=False)
                    
                elif user_input in discussion:
                    message("Let us start with your inputs.Please describe more about your problem statement ?", is_user=False)
                    
                elif user_input in bot_info:
                    message("I am Bot. Your virtual learning assistant ?", is_user=False)
                    
                elif user_input in end_convo:
                    message("Please use respectful words", is_user=False)
    
                elif len(user_input) >=30:
                    message(chatbot_response(user_input),is_user=False) 
                                      
                   
                else:
                    message("Please re-phrase your query to help us understand your issue better", is_user=False)                     
                                       
                    
                    
            else:
                st.warning("Please enter your message to start the conversation")             
             

                    
            
