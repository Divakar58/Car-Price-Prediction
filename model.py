# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#from IPython import get_ipython

# %%
# importing required libraries
import pandas as pd

# %% [markdown]
# ### Data source https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv

# %%
#reading the file
car_df=pd.read_csv(r'car data.csv')


# %%
#previewing the data
car_df.head()

# %% [markdown]
# ### Column dictionary
# 
# - Car Name: name of the car.
# 
# - Year: This column should be filled with the year in which the car was bought.
# 
# - Selling_Price: This column should be filled with the price the owner wants to sell the car at.
# 
# - Present_Price: This is the current ex-showroom price of the car.
# 
# - Kms_Driven: This is the distance completed by the car in km.
# 
# - Fuel_Type: Fuel type of the car.
# 
# - Seller_Type: Defines whether the seller is a dealer or an individual.
# 
# - Transmission: Defines whether the car is manual or automatic.
# 
# - Owner: Defines the number of owners the car has previously had.

# %%
# check for missing values
car_df.isnull().sum()


# %%
# No missing values


# %%
car_df.describe()


# %%
car_df.info()


# %%
#list of columns in dataset
car_df.columns


# %%
cat_variables=['Fuel_Type','Seller_Type','Transmission']


# %%
#subcategories of catergorical column
for i in cat_variables:
    print("column Name ->",i, "\nUnique values ->",car_df[i].unique())


# %%
# converting categorical variables to numerical using binary encoding
car_df['Seller_Type']=car_df['Seller_Type'].map({'Dealer':0,'Individual':1})
car_df['Transmission']=car_df['Transmission'].map({'Manual':0,'Automatic':1})


# %%
# converting categorical variables to numerical using one hot encoding
temp=pd.get_dummies(car_df['Fuel_Type'],drop_first=True)


# %%
car_df=pd.concat([car_df,temp],axis=1)


# %%
car_df.head()


# %%
# droping Fuel_type and Car_Name columns 
car_df.drop(['Car_Name','Fuel_Type'],axis=1,inplace=True)

# %% [markdown]
# #### Derived columns

# %%
# derived no years since car brought
car_df['No.years']=car_df['Year'].apply(lambda x: 2020-x)


# %%
car_df.head()


# %%
car_df.drop('Year',axis=1,inplace=True)


# %%
car_df.head()


# %%
car_df.corr()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# %%
plt.figure(figsize=(10,6))
fig=sns.heatmap(car_df.corr(),annot=True,cmap='RdYlGn')
bottom, top = fig.get_ylim()
fig.set_ylim(bottom + 0.5, top - 0.5)


# %%
X=car_df.iloc[:,1:]
y=car_df.iloc[:,0]


# %%
from sklearn.model_selection import train_test_split


# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# %%
rfg=RandomForestRegressor()


# %%
rfg.fit(X_train,y_train)


# %%
y_pred=rfg.predict(X_test)


# %%
residual=y_test-y_pred
sns.distplot(y_pred)


# %%
from sklearn.metrics import mean_squared_error


# %%
mean_squared_error(y_pred,y_test)


# %%
lin=LinearRegression()


# %%
lin.fit(X_train,y_train)


# %%
y_pred=lin.predict(X_test)


# %%
mean_squared_error(y_pred,y_test)


# %%
xgb=XGBRegressor(objective='reg:squarederror')


# %%
xgb.fit(X_train,y_train)


# %%
y_pred=xgb.predict(X_test)


# %%
mean_squared_error(y_test,y_pred)


# %%
## Hypertuning XGboost paramas using randomsearchCV


# %%
from sklearn.model_selection import RandomizedSearchCV


# %%
# choosing hyperparamas
params={'learning_rate':[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],'n_estimators':[int(i) for i in np.linspace(100,1200,num=12)],'subsample':[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}


# %%
#instance of RandomizedsearchCV
rsc=RandomizedSearchCV(xgb,param_distributions=params,verbose=2,cv=5,scoring='neg_mean_squared_error',return_train_score=True)


# %%
#fit the model
rsc.fit(X_train,y_train)


# %%
rsc.best_params_


# %%
rsc.best_estimator_


# %%
y_pred=rsc.estimator.predict(X_test)


# %%
#Residual analysis to validate the model
sns.distplot(y_pred-y_test)


# %%
sns.scatterplot(y_pred,y_test)


# %%
mean_squared_error(y_pred,y_test)


# %%
import pickle


# %%
with open('model.pkl','wb') as f:
    pickle.dump(rsc.estimator,f)


# %%


