# Basic & visualization modules

#import packages/functions:

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn modules
from sklearn.model_selection import train_test_split                    # & test split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree  import DecisionTreeClassifier         # decision tree
from sklearn.ensemble import AdaBoostClassifier        # adaboosting
from sklearn.neighbors import KNeighborsClassifier      # knn
from sklearn.ensemble import RandomForestClassifier      # random forest
from sklearn.linear_model import LogisticRegression     # logistic regression
from sklearn.ensemble import GradientBoostingClassifier # gbm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#from imblearn.over_sampling import SMOTE


#read in dataset:
data = pd.read_csv('C:/Users/Simon Hofer/OneDrive/Dokumente/Master/Semesterverzeichnis/Semester 1/SeminarIBA/preprocessed_data.csv', sep=',')

#preprocess data (drop, filter,etc.)
data.drop(['ID', 'Customer_ID'], axis=1, inplace=True)

#
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


clean_dataset(data)




#hier schauen ob wir überhaupt ordinal encoden sollen?
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
for column in data.columns:
  if data[column].dtypes == 'object':
    data[column] = ordinal_encoder.fit_transform(data[[column]])


data.Annual_Income = (data.Annual_Income - data.Annual_Income.mean())/data.Annual_Income.std()
data.Monthly_Inhand_Salary = (data.Monthly_Inhand_Salary - data.Monthly_Inhand_Salary.mean())/data.Monthly_Inhand_Salary.std()
data.Type_of_Loan = (data.Type_of_Loan - data.Type_of_Loan.mean())/data.Type_of_Loan.std()
data.Outstanding_Debt = (data.Outstanding_Debt - data.Outstanding_Debt.mean())/data.Outstanding_Debt.std()
data.Credit_History_Age = (data.Credit_History_Age - data.Credit_History_Age.mean())/data.Credit_History_Age.std()
data.Total_EMI_per_month = (data.Total_EMI_per_month - data.Total_EMI_per_month.mean())/data.Total_EMI_per_month.std()
data.Amount_invested_monthly = (data.Amount_invested_monthly - data.Amount_invested_monthly.mean())/data.Amount_invested_monthly.std()




print(data.head())



