from job_scraper import job_urls
from job_scraper import get_job_description
import text_processing
import nltk
import pandas as pd
from text_processing import _removeNonAscii
from text_processing import make_lower_case
from text_processing import remove_stop_words
from text_processing import remove_punctuation
from text_processing import remove_html
from text_processing import stem_sentences
from job_recommender import recommender
import numpy as np

# call function to get job links
"""
job_links, jobs_df = job_urls('https://www.workpac.com/jobs')

# make lower case   
jobs_df['job_title'] = jobs_df['job_title'].apply(make_lower_case)

# call function to get all job descriptions from the job_links list
job_description = []
for link in job_links:
    job_description.append(get_job_description(link))

# convert lists into pandas Series objects
job_links = pd.DataFrame(job_links)
job_description = pd.DataFrame(job_description)

# create df for NLP model

job_data_df = pd.concat([job_links, job_description], axis = 1)
job_data_df = pd.concat([job_data_df, jobs_df['job_title']], axis = 1)
job_data_df.columns=['job_link', 'job_description', 'job_title']

# write to file 

job_data_df.to_csv('job_descriptions.csv')

# NLTK

# nltk.download()
# then download stopwords corpa
"""
# import data
workpac_data = pd.read_csv('job_descriptions.csv')

# make a backup
workpac_data_original = workpac_data

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

recommended_jobs = recommender(model_df)

recommended_jobs = pd.merge(left=recommended_jobs, right=jobs_df, left_on='job_link', right_on='links')
"""
recommended_jobs['candidate_ranking'] = 0

recommended_jobs.loc[(recommended_jobs['cosine_similarity'] > 0.009),'candidate_ranking'] = 'Strong'
recommended_jobs.loc[(recommended_jobs['cosine_similarity'] < 0.009) & 
(recommended_jobs['cosine_similarity'] > 0.005),'candidate_ranking'] = 'Medium'
recommended_jobs.loc[(recommended_jobs['cosine_similarity'] < 0.005),'candidate_ranking'] = 'Low'
"""
recommender_summary = recommended_jobs[['job_title', 'job_link']]

recommender_summary


