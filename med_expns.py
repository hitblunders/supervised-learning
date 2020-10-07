# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Import the dataset 
# Medical Cost Personal Dataset
dataset = pd.read_csv('/kaggle/input/insurance/insurance.csv')

dataset.head()

dataset.isnull().sum()

# Encode categorical data
# Create instance
encode = LabelEncoder()

# Gender of the benificiary
encode.fit(dataset.sex.drop_duplicates()) 
dataset.sex = encode.transform(dataset.sex)

# Is the benificiary smoker or non-smoker
encode.fit(dataset.smoker.drop_duplicates()) 
dataset.smoker = encode.transform(dataset.smoker)

# Residence of benificiary
encode.fit(dataset.region.drop_duplicates()) 
dataset.region = encode.transform(dataset.region)

dataset.head()

dataset.corr()['charges'].sort_values()
f, ax = plt.subplots(figsize=(10, 8))
corr = dataset.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='Greens',square=True, ax=ax)
            
plt.figure(figsize=(12,7))
plt.title("Distribution")
sns.distplot(dataset,color='r')   

sns.pairplot(dataset)

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pastel", data=dataset)

plt.figure(figsize=(12,7))
plt.title("BMI")
sns.distplot(dataset["bmi"], color = 'c')

plt.figure(figsize=(12,7))
plt.title("Children")
sns.distplot(dataset["children"], color = 'r')



plt.figure(figsize=(12,7))
plt.title("Age")
sns.distplot(dataset["age"], color = 'g')

sns.catplot(x="children", kind="count", palette="rainbow", data=dataset)

# Get feature matrix and label vector
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Train the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)

print(regressor.score(X_test,y_test))

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(X_train, y_train)
dtr_y_pred = dtr.predict(X_test)

print('MSE test data: %.3f' % (mean_squared_error(y_test,dtr_y_pred)))
print('R2 test data: %.3f' % (r2_score(y_test,dtr_y_pred)))

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfr.fit(X_train, y_train)
rfr_y_pred = rfr.predict(X_test)

print('MSE test data: %.3f' % (mean_squared_error(y_test,rfr_y_pred)))
print('R2 test data: %.3f' % (r2_score(y_test,rfr_y_pred)))

plt.figure(figsize=(12,7))
plt.title("FINAL PREDICTION")
plt.scatter(rfr.predict(X_train),rfr.predict(X_train)- y_train,c = 'red', marker = 'o', s = 30, alpha = 0.6,label = 'Train data')
plt.scatter(rfr_y_pred,rfr_y_pred - y_test,c = 'cyan', marker = 'o', s = 30, alpha = 0.8,label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend()
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'black')
plt.show()
