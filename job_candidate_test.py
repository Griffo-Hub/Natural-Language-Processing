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
from recommender import recommender
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


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


key_words = input()

# clean input
key_words = _removeNonAscii(key_words)
key_words = make_lower_case(key_words)
key_words = remove_stop_words(key_words)
key_words = remove_punctuation(key_words)
key_words = remove_html(key_words)

# stem input

key_words = stem_sentences(key_words)

# add input to the model_df to calculate tfidf of input against existing data
model_df.loc[-1] = key_words
model_df.index = model_df.index + 1
model_df = model_df.sort_index()

# construct tf vectoriser
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english')

# fit tf
tfidf_matrix = tf.fit_transform(model_df['job_text_cleaned'])

# recommender
a1 = tfidf_matrix.getrow(0).toarray()
dist_list = []
for i in range(tfidf_matrix.shape[0]):
    a2 = tfidf_matrix.getrow(i).toarray()
    dist = cosine_similarity(a1, a2)
    dist_list.append(dist)

model_df['cosine_similarity'] = dist_list

recommended_jobs = model_df.sort_values('cosine_similarity', ascending=False)

# remove cv from df
recommended_jobs = recommended_jobs.iloc[1:]


# pretend candidate_id which will be passed as an argument to the function

candidate_id = 'c004'

recommended_jobs['cadidate_id'] = candidate_id

recommended_jobs.columns = ['job_link', 'job_description', 
                            'cosine_similarity', 'candidate_id']

# write results to file
recommended_jobs.to_csv('job_candidate_similarity.csv', mode='a', header=False)

