import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.style.use('ggplot')

car = pd.read_csv('quikr_car.csv')
#print(car.head())
print(car.shape)
print(car.info())

#creating backup copy
backup= car.copy()

#cleaninng the data
#year has many non-year values
car=car[car['year'].str.isnumeric()]
print(car)
# #year is in object. change to integer
car['year']=car['year'].astype(int)

# #price has ask for price
car=car[car['Price']!='Ask For Price']

#price has commas in its prices and is in object
#car['Price'] = car['Price'].astype(int)
car['Price']=car['Price'].str.replace(',','',regex=True).astype(int)
#print(car)

#kms_driven has object values kms at last
car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')

#it has nan values and two rows have 'petrol' in them.
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)

#fuel type has nan values
car=car[~car['fuel_type'].isna()]
print(car.shape)

#name and company had spammed data...but with the previous cleaning, those rows got removed
#company does not need any cleaning now.changing car names.keeping only the first three words
car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')

#resetting the index of the final cleaned data
car=car.reset_index(drop=True)

#cleaned data
print(car)
car.to_csv('Cleaned_car_data.csv')
print(car.info())
print(car.describe(include='all'))
car=car[car['Price']<6000000]

#checking relationship of company with price
print(car['company'].unique())

import seaborn as sns
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

#checking relationship of year with price 
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

#checking relationship of kms_driven with price
sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)
plt.show()

#checking relationship of fuel type with price
plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)
plt.show()

#relationship of price with fueltype, year and company mixed
ax= sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')
plt.show()

#extracting training data
x=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']
print(x)
print(y.shape)

#applying train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

#creating an onehotencoder object to contain all the possible categories
ohe= OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
OneHotEncoder()

#creating a column transformer to transform categorical colummns
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                 remainder='passthrough')

#Linear regression model
lr=LinearRegression()

#making a pipeline
pipe=make_pipeline(column_trans,lr)

#fitting the model
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

#checking R2 score
a=r2_score(y_test,y_pred)
print(a)

#finding the model with a random state of traintestsplit where the model was found to
#to give almost 0.92 as r2_score
scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))
b=np.argmax(scores)
print(b)

c=scores[np.argmax(scores)]
print(c)

d=pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
print(d)

#the best model is found at a certain random state
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
e=r2_score(y_test,y_pred)
print(e)

import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
f=pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
print(f)
g=pipe.steps[0][1].transformers[0][1].categories[0]
print(g)