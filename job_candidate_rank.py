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
import plotly.graph_objects as go

def app():

    ##### Set up data for modelling

    # import data to model CV against
    workpac_data = pd.read_csv('job_descriptions.csv')

    # import job_candidate data for candidate ranking feature
    job_candidate_rank_df = pd.read_csv('job_candidate_similarity.csv')
    candidates = pd.read_csv('candidates.csv')

    # create table with just data needed for this app

    job_candidate_rank_df = pd.merge(left=job_candidate_rank_df, right=workpac_data, left_on='job_link', right_on='job_link')
    job_candidate_rank_df = pd.merge(left=job_candidate_rank_df, right=candidates, left_on='candidate_id', right_on='candidate_id')


    job_candidate_rank_df = job_candidate_rank_df[['job_title', 'job_link', 'candidate_id', 'candidate_name', 'cosine_similarity']]

    jobs = job_candidate_rank_df['job_title'] + ' ' + job_candidate_rank_df['job_link']
    
    # create another column for filtering the data frame for the output
    job_candidate_rank_df['concat_title_link'] = job_candidate_rank_df['job_title'] + ' ' + job_candidate_rank_df['job_link']

    # title of streamlit app
    st.title('Candidate Ranking System Using Natural Language Processing (NLP)')

    # sidebar
    st.sidebar.subheader('Job Selector')

    # select box
    selected_job = st.sidebar.selectbox(
        'Please select a Job',
        (jobs)
    )

    # filtered candidates

    ranked_candidates = job_candidate_rank_df.loc[(job_candidate_rank_df.concat_title_link == selected_job)]
    ranked_candidates = ranked_candidates.filter(['candidate_id', 'candidate_name', 'cosine_similarity'])
    ranked_candidates = ranked_candidates.sort_values('cosine_similarity', ascending=False)
    st.write(ranked_candidates)



