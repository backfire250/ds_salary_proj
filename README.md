# Data Science Salary Estimator: Project Overview 
**Project Goal**:  Created a tool that estimates data science salaries based on some factors associated with the job: geography, job title, company size, industry, # of competitors etc.
* Scraped 1000 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on Python, Excel, AWS, and Spark. 
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
* Built a client facing API using Flask

## Project Steps
1. Project Planning
2. Data Collection
3. Data Cleaning
4. Exploratory Data Analysis
5. Model Building
6. Production

## Code and Resources Used  
**Python Version:** 3.7   
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**Spyder IDE**  
**Jupyter Notebooks**  
**GridsearchCV**    
**For Web Framework Requirements:**  `pip install -r requirements.txt`  
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c905  
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2=

## YouTube Project Walk-Through
For this project, I followed the below YouTube video series to familiarize myself with some data science tools:
https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Web Scraping
Tweaked the web scraper github repo (above) to scrape 1000 job postings from glassdoor.com. With each job posting, I got the following:

* Job Title
* Salary Estimate
* Job Description
* Rating
* Company
* Location
* Company Headquarters
* Company Size
* Company Founded Date
* Type of Ownership
* Industry
* Sector
* Revenue  
* Competitors

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows from data set where salary was not listed 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column to indicate if the job was at the company’s headquarters 
*	Transformed 'founded date' column into 'age of company' 
*	Made columns to show if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Added column for simplified job title and seniority level 
*	Column for job description length

## EDA (Exploratory Data Analysis)
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.

![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/correlation_viz.png)
![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/pos_by_state.png)
![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/salary_by_job_title.PNG)

## Model Building

First, I transformed the categorical variables into dummy variables. I also split the data into train and test sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error (the average variance between actual values and projected values in the dataset). I chose MAE because it is relatively easy to interpret and the outliers weren't particularly bad for this type of model.

I tried three different models:
*    **Multiple Linear Regression** - Used as the baseline for my model.
*    **Lasso Regression** - Because of the sparse data from the many categorical variables, I thought a normalized regression would be effective.
*    **Random Forest** - Again, with the sparsity of the available data, I thought that this would be a good fit.

## Model Performance
The Random Forest model far outperformed the other approaches on the test and validation sets.
*    **Random Forest** : MAE = 11.22
*    **Linear Regression** : MAE = 18.86
*    **Ridge Regression** : MAE = 19.67

## Productionization
In this step, I built a Flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the resource section above. The API endpoint takes in a request with a list of values from a job posting and returns an estimated salary.
