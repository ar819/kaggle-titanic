import pandas as pd
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Check if GPU is available
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
device = torch.device('cuda:0')

# Read the data

train_df = pd.read_csv('spaceship-titanic/train.csv')
X_train = train_df.drop('Transported', axis=1)
y_train = train_df['Transported']

# split x train into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



test_df = pd.read_csv('spaceship-titanic/test.csv')
X_test = test_df

# Check the data
print(train_df.head())

# Check the data types
print(train_df.info())

# Check the missing values
print(train_df.isnull().sum())

# Check no. unique values in the columns along with total count along with column names
print(pd.concat([train_df.nunique(), train_df.count(), train_df.isnull().sum()], axis=1, keys=['Unique Values', 'Count', 'Missing Values']))

# One hot encode the categorical columns
# Define the columns to be one hot encoded
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Define the pipeline
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

# Fit the preprocessor
preprocessor.fit(X_train)
# Transform the data
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
X_val = preprocessor.transform(X_val)

X_train

from sklearn.ensemble import GradientBoostingClassifier


# training loop to find the best hyperparameters
learning_rate = [0.25,0.5,0.75,1]
n_estimators = [500,1000]
max_depth = [1] 

estimates = []


for lr in learning_rate:
    for ne in n_estimators:
        for md in max_depth:
            gbc = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, max_depth=md, random_state=0, subsample=0.5)
            gbc.fit(X_train, y_train)
            print(f'Learning Rate: {lr}, n_estimators: {ne}, max_depth: {md}')
            print(gbc.score(X_train, y_train))
            print(gbc.score(X_val, y_val))
            estimates.append((lr, ne, md, gbc.score(X_train, y_train), gbc.score(X_val, y_val)))    

# graph of the results
import matplotlib.pyplot as plt

estimates_df = pd.DataFrame(estimates, columns=['Learning Rate', 'n_estimators', 'max_depth', 'Train Accuracy', 'Validation Accuracy'])
estimates_df['Validation Accuracy'] = estimates_df['Validation Accuracy'] * 100
estimates_df['Train Accuracy'] = estimates_df['Train Accuracy'] * 100
estimates_df = estimates_df.melt(id_vars=['Learning Rate', 'n_estimators', 'max_depth'], value_vars=['Train Accuracy', 'Validation Accuracy'], var_name='Accuracy', value_name='Score')
estimates_df['Accuracy'] = estimates_df['Accuracy'].str.replace(' Accuracy', '')
estimates_df['Score'] = estimates_df['Score'].astype(int)

plt.figure(figsize=(12, 8))
plt.title('Gradient Boosting Classifier Hyperparameter Tuning')
plt.xlabel('Hyperparameter')
plt.ylabel('Score')

for lr in learning_rate:
    for ne in n_estimators:
        for md in max_depth:
            subset = estimates_df[(estimates_df['Learning Rate'] == lr) & (estimates_df['n_estimators'] == ne) & (estimates_df['max_depth'] == md)]
            plt.plot(subset['Accuracy'], subset['Score'], label=f'LR: {lr}, NE: {ne}, MD: {md}')

plt.legend()
plt.show()

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5, max_depth=1, random_state=0)
gbc.fit(X_train, y_train)
# Predict the target
y_pred = gbc.predict(X_test)

# Pd data frame submission
submission = test_df['PassengerId'].to_frame()
submission['Transported'] = y_pred
submission.to_csv('spaceship-titanic/submission.csv', index=False)

