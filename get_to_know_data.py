import numpy as np
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


#### START ###############################################IBA SEMINAR DATA PREPARATION########################################

# Reading data from a csv file
data = pd.read_csv("preprocessed_data.csv")

# Preprocessing wie Lars

data.drop(['ID', 'Customer_ID'], axis=1, inplace=True)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
for column in data.columns:
  if data[column].dtypes == 'object':
    data[column] = ordinal_encoder.fit_transform(data[[column]])

data = clean_dataset(data)
data = data.reset_index()

#TODO: alle Werte die keinen Sinn machen rauswerfen

#Datenaufbereitung

print("\n", "********Randinformationen********", "\n")
print("Es werden Kundendaten betrachtet. Der Datensatz umfasst", data.shape[0], "Kunden. Jeder Kunde hat", data.shape[1], "Attribute", "\n")


print("\n","********Betrachtung der Attribute********", "\n")
print("Die Attribute lauten: ", ', '.join(data.columns), "\n", "\n")

print("Im folgenden werden Minimum, Maximum und der Average der einzlenen Attribute ausgegeben""\n")
data.loc[data.shape[0]] = data.min()
data.loc[data.shape[0]] = data.max()
data.loc[data.shape[0]] = data.mean()

print("***** Minimum : ***** ", "\n", data.loc[data.shape[0]-3], "\n")
print("***** Maximum : ***** ", "\n",data.loc[data.shape[0]-2], "\n")
print("***** Average : ***** ", "\n",data.loc[data.shape[0]-1], "\n")

plt.figure()
data["Age"].plot.hist(bins=9, range=(0,60), alpha=0.5, color = "blue", edgecolor="black")
plt.xlabel("Age")
plt.tight_layout()
plt.ylabel("Frequency of occurence")
plt.show()

plt.figure()
data["Annual_Income"].plot.hist(bins=12, range=(data["Annual_Income"].min(), 200000), alpha=0.5, color = "red", edgecolor="black")
plt.xlabel("Annual Income")
plt.tight_layout()
plt.ylabel("Frequency of occurence")
plt.show()

#### END ###############################################IBA SEMINAR DATA PREPARATION########################################
