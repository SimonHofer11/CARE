import numpy as np
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


#### START ###############################################IBA SEMINAR DATA PREPARATION########################################

# Reading data from a csv file
data = pd.read_csv("prepreprocessed_data.csv")
data.reset_index(drop=True, inplace=True)
data.drop(['Type_of_Loan'], axis=1, inplace=True)

print("\n", "********Informationen Datensatz********", "\n")
print("Es werden Kundendaten betrachtet. Der Datensatz umfasst", data.shape[0], "Kunden. Jeder Kunde hat", data.shape[1], "Attribute", "\n")


# Preprocessing

#drop unlogicial rows

print("Zunächst werden alle Kunden die unlogische Attribute (z.B. Alter = -3) enthalten und Kunden mit fehlenden Attributen aus dem Datensatz entfernt.","\n")
data = data[data['Num_Bank_Accounts'] >= 0]
data = data[data['Num_of_Loan'] >= 0]
data = data[data['Delay_from_due_date'] >= 0]
data = data[data['Num_of_Delayed_Payment'] >= 0]
data = data[data['Changed_Credit_Limit'] >= 0]
## Handling missing values
data = data.dropna().reset_index(drop=True)
print("Außerdem werden 3 Spalten (ID, Customer ID und SNN), die keinen Informationswert enthalten aus dem Datzsatz entfernt." "\n")
# delete some columns
data.drop(['ID', 'Customer_ID', 'SSN'], axis=1, inplace=True)
print("Danach umfasst der Datensatz noch", data.shape[0], "Kunden. Jeder Kunde hat jetzt nur noch", data.shape[1], "Attribute", "\n")


#Aufteilung in discret und continous
continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                       'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                       'Changed_Credit_Limit',
                       'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                       'Amount_invested_monthly', 'Monthly_Balance']
#TODO: Type of Loans
discrete_features = ['Month', 'Occupation', 'Credit_Mix',
                     'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour']

print("Von den übrig gebliebenen Attribute sind ", len(continuous_features), "Attribute stetig und ", len(discrete_features), "diskret")
print("\n")
print("\n")

print("\n","********Betrachtung der Attribute********", "\n")
print("Die Attribute lauten: ",  "\n", ', '.join(data.columns), "\n", "\n")

print("Im folgenden werden Minimum, Maximum und der Average der stetigen Attribute ausgegeben""\n")

print("Die stetigen Attribute lauten:", "\n" , continuous_features)

print("***** Minimum : ***** ", "\n", data[continuous_features].min(), "\n")
print("***** Maximum : ***** ", "\n", data[continuous_features].max(), "\n")
print("***** Average : ***** ", "\n", data[continuous_features].mean(), "\n")
print("\n")

print("Die discreten Attribute lauten:" , discrete_features)
print("\n")

print("Bei Occupation gibt es die Auswahl zwischen: ", data["Occupation"].unique())
print("\n")
print("Bei Credit_Mix gibt es die Auswahl zwischen: ", data["Credit_Mix"].unique())
print("\n")
print("Bei Credit_History_Age gibt es die Auswahl zwischen: ", data["Credit_History_Age"].unique())
print("\n")
print("Bei Payment_of_Min_Amount gibt es die Auswahl zwischen: ", data["Payment_of_Min_Amount"].unique())
print("\n")
print("Bei Payment_Behaviour gibt es die Auswahl zwischen: ", data["Payment_Behaviour"].unique())

for attribut in continuous_features:
    plt.figure()
    data[attribut].plot.hist(bins=9, range=(data[attribut].min()*0.9, data[attribut].max()*1.1), alpha=0.5, color = "blue", edgecolor="black")
    plt.xlabel(attribut)
    plt.tight_layout()
    plt.ylabel("Frequency of occurence")
    plt.savefig(attribut)


for attribut in discrete_features:
    plt.figure()
    data[attribut].value_counts().plot(kind='bar', color='red', alpha=0.5, edgecolor='black')
    plt.xlabel(attribut)
    plt.ylabel("Frequency of occurrence")
    plt.tight_layout()
    plt.savefig(attribut)

#### END ###############################################IBA SEMINAR DATA PREPARATION########################################
