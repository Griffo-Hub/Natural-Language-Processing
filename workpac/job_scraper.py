import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from text_processing import make_lower_case

# constructors

options = Options()
options.add_argument("start-maximized")
executable_path = 'C:/Users/david.griffith/OneDrive/Code/Seek_Recommender/chromedriver.exe'
driver = webdriver.Chrome(chrome_options=options, executable_path=executable_path)

# function to scrape all job urls from workpac and store in a list

def job_urls(url):
    
    # initialise a list to store all workpac jobs into

    job_links = []
    job_title = []

    # loop through all search results using pagination

    while url:
        
        # get soup
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # use selenium to get client side JavaScript text
        driver.get(url)

        # scrape job link and job title using selenium
        elems = driver.find_elements_by_css_selector(".job-title [href]")
        titles = driver.find_elements_by_css_selector(".job-title")
        
        # store urls in list
        links = [elem.get_attribute('href') for elem in elems]
        job_links += links

        # store job title in list
        titles = [title.text for title in titles]
        job_title += titles
        # next page url
        
        url = soup.findAll('a', {'rel': 'next'})
        if url:
            url = 'https://www.workpac.com' + url[0].get('href')
        else:
            break
    
    # create dictionary for job links and job titles and then convert to data frame

        dict = {'links': job_links, 'job_title': job_title}

        jobs_df = pd.DataFrame(dict)

    return job_links, jobs_df

# function to scrape all job descriptions for each job link

def get_job_description(url):

    # get soup
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    # get job description soup
    results = soup.find('div', {'class':'job col-xl-8 col-lg-12 col-md-12 clearfix left'})
    
    # loop through and get all text in the job article
    job_description = [e.get_text() for e in results.find_all('article', {'class': 'description'})]
    job_description = job_description[0]
    job_description = clean_text(job_description)

    return job_description

# funciton used to remove tabs and carriage returns in the job description

def clean_text(text):
    for ch in ['\n','\t']:
        if ch in text:
            text = text.replace(ch, ' ')
    return text
