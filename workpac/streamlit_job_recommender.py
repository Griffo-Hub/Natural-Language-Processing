import streamlit as st
import pandas as pd 
import plotly_express as px
from docx2python import docx2python
from docx2python.iterators import enum_paragraphs
import numpy as np
from text_processing import _removeNonAscii
from text_processing import make_lower_case
from text_processing import remove_stop_words
from text_processing import remove_punctuation
from text_processing import remove_html
from text_processing import stem_sentences
from recommender import recommender

##### Set up data for modelling

# import data to model CV against
workpac_data = pd.read_csv('workpac/job_descriptions.csv')

# remove id column as we can use index
workpac_data = workpac_data[['job_link','job_description', 'job_title']]

# clean Na's so functions can run
workpac_data = workpac_data.replace(np.nan, ' ', regex=True)

# clean job description
workpac_data['job_description_cleaned'] = workpac_data['job_description'].apply(_removeNonAscii)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func = make_lower_case)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func = remove_stop_words)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func=remove_punctuation)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func=remove_html)

workpac_data['job_title_cleaned'] = workpac_data['job_title'].apply(_removeNonAscii)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func = make_lower_case)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func = remove_stop_words)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func=remove_punctuation)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func=remove_html)


# stem cleaned data
workpac_data['job_description_cleaned'] = workpac_data['job_description_cleaned'].apply(stem_sentences)
workpac_data['job_title_cleaned'] = workpac_data['job_title_cleaned'].apply(stem_sentences)

workpac_data['job_text_cleaned'] = workpac_data['job_title_cleaned'] + ' ' + workpac_data['job_description_cleaned']

model_df = workpac_data[['job_link', 'job_text_cleaned']]

# title of streamlit app
st.title('Candidate Job Ranking System Using Natural Language Processing (NLP)')

# sidebar
st.sidebar.subheader('Files')

# file uploader
uploaded_file = st.sidebar.file_uploader(label='Please upload your CV', type=['docx'])

# create message variable
message = ''

if uploaded_file is not None:
    try:
        doc = docx2python(uploaded_file).text
        message = 'File upload successful'
    except Exception as e:
        message = 'There was an error uploading your file'

# display success/failure message

st.sidebar.text(message)

# display uploaded file (text only as the method used in docx2python is .text)
st.text(doc)

# call recommender function
recommended_jobs = recommender(model_df, doc)

# create output dataframe
recommended_jobs = pd.merge(left=recommended_jobs, right=workpac_data, left_on='job_link', right_on='job_link')

# summarise output
recommender_summary = recommended_jobs[['job_title', 'job_link']]

# display output
st.write(recommender_summary)

