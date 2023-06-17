
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler

def PrepareIBA_dataset(path_param,name_param):
    dataset_name = "preprocessed_data.csv"

    ## Reading data from a csv file
    df = pd.read_csv('prepreprocessed_data.csv',sep=',')

    #drop unrealistic rows
    df = df[df['Num_Bank_Accounts'] >= 0]
    df = df[df['Num_of_Loan'] >= 0]
    df = df[df['Delay_from_due_date'] >= 0]
    df = df[df['Num_of_Delayed_Payment'] >= 0]
    df = df[df['Changed_Credit_Limit'] >= 0]

    #Löscht mehrere Spalten
    df.drop(['ID', 'Customer_ID', 'SSN','Type_of_Loan','Month'], axis=1, inplace=True)

    #Umwandeln der Credit History Age Variable
    df['Credit_History_Age'] = df['Credit_History_Age'].str.extract(r'(\d+)')


    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'Credit_Score'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]
    print("df_y_head: ",df_y[0:5])



    #ohne Targetvariable
    continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                           'Interest_Rate','Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit',
                           'Num_Credit_Inquiries','Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
                           'Amount_invested_monthly','Monthly_Balance']

    #discrete_features = ['Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix',
    #                     'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour']


    discrete_features = ['Occupation', 'Credit_Mix',
                     'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour','Auto Loan',' Credit-Builder Loan',
                         ' Personal Loan', 'Debt Consolidation Loan','Home Equity Loan','Mortgage Loan','Payday Loan','Student Loan',
                         'Not Specified']
    x = len(continuous_features) + len(discrete_features)
    print(x)

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': "IBA_seminar_dataset",
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }
    print("here we go")

    return dataset



import numpy as np

def ord2ohe(X_ord, dataset):
    continuous_availability = dataset['continuous_availability']
    discrete_availability = dataset['discrete_availability']
    ohe_feature_encoder = dataset['ohe_feature_encoder']
    len_continuous_ord = dataset['len_continuous_ord']
    len_discrete_ord = dataset['len_discrete_ord']

    if X_ord.shape.__len__() == 1:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete.reshape(1,-1)).ravel()
            X_ohe = np.r_[X_continuous, X_discrete]
            return X_ohe
        elif continuous_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_ohe = X_continuous.copy()
            return X_ohe
        elif discrete_availability:
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete.reshape(1, -1)).ravel()
            X_ohe = X_discrete.copy()
            return X_ohe
    else:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete)
            X_ohe = np.c_[X_continuous,X_discrete]
            return X_ohe
        elif continuous_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_ohe = X_continuous.copy()
            return X_ohe
        elif discrete_availability:
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete)
            X_ohe = X_discrete.copy()
            return X_ohe


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Defining path of data sets and experiment results
path = './'
dataset_path = path + 'datasets/'

# Defining the list of data sets
datasets_list = {
    'IBA_seminar_dataset': ("preprocessed_data.csv", PrepareIBA_dataset, 'classification'),
}

# Defining the list of black-boxes
blackbox_list = {
    'rf-c': RandomForestClassifier,
    # 'nn-c': MLPClassifier,
    # 'gb-c': GradientBoostingClassifier
}

for dataset_kw in datasets_list:
    print('dataset=', dataset_kw)
    print('\n')

    # Reading a data set
    dataset_name, prepare_dataset_fn, task = datasets_list[dataset_kw]
    dataset = prepare_dataset_fn(dataset_path, dataset_name)


X, y = dataset['X_ord'], dataset['y']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=45)

X_train_ohe = ord2ohe(X_train, dataset)
X_test_ohe = ord2ohe(X_test, dataset)

blackbox = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=None, min_samples_leaf=1, min_samples_split=2)
blackbox.fit(X_train_ohe, Y_train)
pred_test = blackbox.predict(X_test_ohe)
explainer = shap.Explainer(blackbox)
shap_values = explainer.shap_values(X_train_ohe)
shap.summary_plot(shap_values, X_train_ohe)
plt.savefig("summary_plot.png")
