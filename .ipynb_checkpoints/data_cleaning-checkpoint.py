# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:41:08 2023

@author: eredfield
"""

import pandas as pd
df = pd.read_csv('glassdoor_jobs.csv')

# create new columns 'hourly' and 'employer_provided' if those words exist in the Salary Estimate column
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)


#get rid of jobs where the salary is listed as $-1:
df = df[df['Salary Estimate'] != '-1']

## salary parsing
# get rid of all the (Glassdoor est.) in the Salary Estimate field
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
# remove the K's and $
minus_kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
# remove the 'per hour' and 'employer provided salary' text from the minus_kd df
minus_hr = minus_kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))
# make a min_salary column out of the number that appears first in a salary range
df['min_salary'] = minus_hr.apply(lambda x: int(x.split('-')[0]))
# make a min_salary column out of the number that appears second in a salary range
df['max_salary'] = minus_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary)/2

## Company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)

## State field
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

## Age of company
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2023 -x)

## Parsing of job description (python, etc.)
#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

#r studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df_out = df.drop(['Unnamed: 0'], axis=1)

df_out.to_csv('salary_data_cleaned.csv', index = False)