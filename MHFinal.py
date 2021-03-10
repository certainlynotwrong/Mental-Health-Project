import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic


# data cleaning part
data = pd.read_csv (r'/Users/billli/Documents/survey(1).csv')
print(data)
flag = (data.mental_health_consequence == 'No')|(data.mental_health_consequence == 'Yes')
data = data[flag]

# Timestamp to days
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m').dt.floor('D')


# Gender
data['Gender'].replace(to_replace = ['Cis Male', 'M', 'Mail', 'Make','Male', 'Male ',  'Male (CIS)', 'Malr', 'Man', 'm', 'maile', 'male',
                                     'male leaning androgynous',  'ostensibly male, unsure what that really means',  'something kinda male?', 'Guy (-ish) ^_^'], value = 1, inplace = True)

data['Gender'].replace(to_replace = [ 'F','Female','Female ', 'Female (cis)', 'Female (trans)', 'Genderqueer',  'Woman',
 'cis-female/femme', 'f', 'femail', 'female',  'queer',], value = 0, inplace = True)

data['Gender'].replace(to_replace = [ 'A little about you', 'All', 'Androgyne', 'Enby', 'M', 'Nah', 'Neuter', 'fluid', 'p'], value = np.nan, inplace = True)
 

# 'mental_health_consequence'
data['mental_health_consequence'].replace(to_replace = ['No'], value = 0, inplace = True)
data['mental_health_consequence'].replace(to_replace = ['Yes'], value = 1, inplace = True)

# 'phys_health_consequence'
data[ 'phys_health_consequence'].replace(to_replace = ['No'], value = 0, inplace = True)
data[ 'phys_health_consequence'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data[ 'phys_health_consequence'].replace(to_replace = ['Maybe'], value = np.nan, inplace = True)

# Age
data[ (data['Age']> 100) | (data['Age']<=0) ] = np.nan


# self employed
data['self_employed'].replace(to_replace = ['No'], value = 0, inplace = True)
data['self_employed'].replace(to_replace = ['Yes'], value = 1, inplace = True)


# family history
data['family_history'].replace(to_replace = ['No'], value = 0, inplace = True)
data['family_history'].replace(to_replace = ['Yes'], value = 1, inplace = True)


# treatment
data['treatment'].replace(to_replace = ['No'], value = 0, inplace = True)
data['treatment'].replace(to_replace = ['Yes'], value = 1, inplace = True)


# work_interfere
data['work_interfere'].replace(to_replace = ['Never'], value = 0, inplace = True)
data['work_interfere'].replace(to_replace = ['Often'], value = 1, inplace = True)
data['work_interfere'].replace(to_replace = ['Rarely'], value = 2, inplace = True)
data['work_interfere'].replace(to_replace = ['Sometimes'], value = 3, inplace = True)


# remote work
data['remote_work'].replace(to_replace = ['No'], value = 0, inplace = True)
data['remote_work'].replace(to_replace = ['Yes'], value = 1, inplace = True)

# tech comapnies
data['tech_company'].replace(to_replace = ['No'], value = 0, inplace = True)
data['tech_company'].replace(to_replace = ['Yes'], value = 1, inplace = True)

# Benefit
data['benefits'].replace(to_replace = ['No'], value = 0, inplace = True)
data['benefits'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['benefits'].replace(to_replace = ["Don't know"], value = np.nan, inplace = True)

# care_options
data['care_options'].replace(to_replace = ['No'], value = 0, inplace = True)
data['care_options'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['care_options'].replace(to_replace = ['Not sure'], value = np.nan, inplace = True)


# wellness program
data['wellness_program'].replace(to_replace = ['No'], value = 0, inplace = True)
data['wellness_program'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['wellness_program'].replace(to_replace = ["Don't know"], value = np.nan, inplace = True)


# seek help
data['seek_help'].replace(to_replace = ['No'], value = 0, inplace = True)
data['seek_help'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['seek_help'].replace(to_replace = ["Don't know"], value = np.nan, inplace = True)


# 'anonymity'
data['anonymity'].replace(to_replace = ['No'], value = 0, inplace = True)
data['anonymity'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['anonymity'].replace(to_replace = ["Don't know"], value = np.nan, inplace = True)

#'leave'
data[ 'leave'].replace(to_replace = ["Don't know"], value = np.nan, inplace = True)

# supervisor
data['supervisor'].replace(to_replace = ['No'], value = 0, inplace = True)
data['supervisor'].replace(to_replace = ['Yes'], value = 1, inplace = True)
data['supervisor'].replace(to_replace = ["Maybe"], value = np.nan, inplace = True)



# To check the unique variables
set(data[ 'mental_health_consequence'])
set(data['Gender'])
set(data['Age'])
set(data['Country'])
set(data['state'])
set(data['self_employed'])
set(data['family_history'])
set(data['treatment'])
set(data['work_interfere'])
set(data['no_employees'])# need more category
set(data['remote_work'])
set(data[ 'tech_company'])
set(data[ 'benefits'])
set(data['care_options'])
set(data[ 'wellness_program'])
set(data[ 'seek_help'])
set(data[ 'anonymity'])
set(data[ 'leave'])
set(data[ 'coworkers'])
set(data[ 'supervisor'])

# irrelevant variabels
set(data[ 'mental_health_interview'])
set(data['phys_health_interview'])      
set(data[ 'mental_vs_physical'])
set(data[ 'obs_consequence'])
set(data[ 'comments'])



print(data.shape)
print(data.columns)
print(data.info())

# Age result
data['Age'].plot.hist()
plt.show()

sns.distplot(data['Age']) # more fancy plot
plt.show()

"""
# how does mental health  vary according to age
data = data.sort_values('Age')
fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.countplot(data = data,x = 'Age',hue= 'treatment')
plt.title('Age vs mental health condition')
ax.set_xticklabels(labels= np.unique(data.loc[:,'Age'].values), rotation=45, ha='right')
"""






# how does mental health  vary according to location
data = data.sort_values('Country')
fig, ax = plt.subplots(figsize = (48,24))    
fig = sns.countplot(data = data,x = 'Country',hue= 'treatment')
plt.title('Country vs mental health condition')
ax.set_xticklabels(labels= np.unique(data.loc[:,'Age'].values), rotation=45, ha='right')


mental_ratio = data.groupby('Age').sum()/ data.groupby('Age').count()
fig, ax = plt.subplots(figsize = (12,6))    
fig = plt.plot(mental_ratio.index, mental_ratio.loc[:,'treatment'])

"""
# how does mental health  vary according to time
fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.countplot(data = data,x = 'Timestamp',hue= 'treatment')
plt.title('Time  vs mental health condition')
x_dates = data['Timestamp'].dt.strftime('%Y-%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
"""

# how does gender  vary according to treatment
mosaic(data, index = ['Gender', 'mental_health_consequence'])

"""
fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.countplot(data = data,x = 'treatment',hue= 'Gender')
plt.title('Gender  vs mental health condition')
x_dates = data['Timestamp'].dt.strftime('%Y-%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
"""
"""
#Anonymity vs Mental Health Condition
fig, ax = plt.subplots(figsize = (12, 6))
fig = sns.countplot(data = data, x = 'treatment',hue = 'anonymity')
plt.title("Anonymity vs Mental health Condition")
x_dates = data['Timestamp'].dt.strftime('%Y-%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
"""
#Mental Health Benefits vs Mental health Condition
"""
fig, ax = plt.subplots(figsize = (12, 6))
fig = sns.countplot(data = data, x = 'treatment',hue = 'benefits')
plt.title("Mental Health Benefits vs Mental health Condition")
x_dates = data['Timestamp'].dt.strftime('%Y-%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
"""
"""
fig, ax = plt.subplots(figsize = (12, 6))
fig = sns.countplot(data = data, x = 'treatment',hue = 'remote_work')
plt.title("Remote Work Availibility  vs Mental health Condition")
x_dates = data['Timestamp'].dt.strftime('%Y-%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
"""

print(data.loc[:, ['anonymity', 'mental_health_consequence']].corr())
# to calculate the correlation between gender and mental health consequence
print(data.loc[:, ['Gender', 'mental_health_consequence']].corr())



data.to_csv('data_cleaned.csv', index=False)





from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
# how many are missing



col_numeric = [cols for cols in data.columns if data[cols].dtype in ['int64', 'float64']]
model_data = data[col_numeric]

model_data = model_data.loc[:, np.sum(np.isnan(model_data),axis=0)/len(model_data)<0.3]

# drop
model_data = model_data.dropna(how='any')

lg = LogisticRegression(random_state=0, solver='lbfgs', fit_intercept=False)

model = lg.fit(model_data.drop('treatment', errors='ignore'), model_data['treatment'])

model.predict(model_data.drop('treatment', errors='ignore'))
model.predict_proba(model_data.drop('treatment', errors='ignore'))
model_data['treatment'].values


print("Test Accuracy: ",accuracy_score(model_data['treatment'].values, model.predict(model_data.drop('treatment', errors='ignore')),normalize=True))


y = model_data['treatment'].values
p_hat = model.predict_proba(model_data.drop('treatment', errors='ignore'))[:,1]

sum(2*( y * np.log(y/p_hat +0.0001) + (1-y)* np.log((1-y)/(1-p_hat) +0.0001)))
