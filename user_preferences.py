from utils import *

def userPreferences(dataset, x_ord):

    x_org = ord2org(x_ord, dataset)

    print('\n')
    print('----- possible values -----')
    for f_val in dataset['feature_values']:
        print(f_val)

    print('\n')
    print('----- instance values -----')
    for i, f in enumerate(dataset['feature_names']):
        print(f+':', x_org[i])

    ## discrete constraints = {'fix', {v1, v2, v3, ...}}
    ## continuous constraints = {'fix', 'l', 'g', 'le', 'ge', [lb, ub]}
    ## constraints = {feature_name_1: (constraint, importance), feature_name_2: (constraint, importance), ...}








    #### START ###############################################IBA SEMINAR DATA PREPARATION########################################
    '''
    if dataset['name'] == 'IBA_seminar_dataset':

        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age) and features with set {} values (e.g., work-class) are discrete

        # {'Month': {0, 7}}
        # {'Age': [14, 60]}
        # {'SSN': {0,12457}}
        # {'Occupation': {0, 15}}
        # {'Annual_Income': [7005.93,277804]}
        # {'Monthly_Inhand_Salary': [303.645417, 14978.33667]}
        # {'Num_Bank_Accounts': [-1, 744]
        # {'Num_Credit_Card': [0, 1000]}
        # {'Interest_Rate': [1, 4998]}
        # {'Num_of_Loan': [-100,1096]}
        # {'Type_of_Loan': [0,6244]}
        # {'Delay_from_due_date': [-5,65]}
        # {'Num_of_Delayed_Payment': [-3,996]}
        # {'Changed_Credit_Limit': [-6.48,29]}
        # {'Num_Credit_Inquiries': [0,2399]}
        # {'Credit_Mix': {0,3}}
        # {'Outstanding_Debt': [0.23, 4949.89]}
        # {'Credit_Utilization_Ratio': [20, 44.992922]}
        # {'Credit_History_Age': {0,403}}
        # {'Payment_of_Min_Amount': {0,2}}
        # {'Total_EMI_per_month': [0,74958]}
        # {'Amount_invested_monthly': [0,10000]}
        # {'Payment_Behaviour': {0,6}}
        # {'Monthly_Balance': [0.00776,1602.040519]}
        # {'Credit_Score': [0,2]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge', 1), #1 steht für importance Strafterm (1-10)
                       'sex': ('fix', 1),
                       'race': ('fix', 1),
                       'native-country': ('fix', 1),
                       'income': ([0 , 1000] , 1)},
        #fixieren: 'fix'
        #greater than or equal to the current value: 'ge'
        #gretaer: 'g'
        #zwischen: '[1,10] / {Kategorische Werte}
        #less than the current value: 'l'
        #less or equal than the current value: 'le'



        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    #### END ###############################################IBA SEMINAR DATA PREPARATION########################################
    '''





    ## Adult data set
    if dataset['name'] == 'adult':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age) and features with set {} values (e.g., work-class) are discrete

        # {'age': [17, 90]}
        # {'fnlwgt': [13769, 1484705]}
        # {'capital-gain': [0, 99999]}
        # {'capital-loss': [0, 4356]}
        # {'hours-per-week': [1, 99]}
        # {'work-class': {' Self-emp-not-inc', ' Local-gov', ' State-gov', ' Self-emp-inc', ' Without-pay',
        #                 ' Federal-gov', ' Private'}}
        # {'education-num': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
        # {'education': {' 12th', ' Some-college', ' Assoc-acdm', ' Doctorate', ' Assoc-voc', ' Prof-school',
        #                ' Preschool', ' HS-grad', ' 7th-8th', ' 1st-4th', ' 9th', ' 10th', ' 11th', ' Masters',
        #                ' Bachelors', ' 5th-6th'}}
        # {'marital-status': {' Married-AF-spouse', ' Divorced', ' Married-civ-spouse', ' Married-spouse-absent',
        #                     ' Widowed', ' Never-married', ' Separated'}}
        # {'occupation': {' Farming-fishing', ' Handlers-cleaners', ' Craft-repair', ' Machine-op-inspct',
        #                 ' Transport-moving', ' Other-service', ' Prof-specialty', ' Armed-Forces', ' Priv-house-serv',
        #                 ' Adm-clerical', ' Exec-managerial', ' Sales', ' Tech-support', ' Protective-serv'}}
        # {'relationship': {' Husband', ' Not-in-family', ' Own-child', ' Other-relative', ' Wife', ' Unmarried'}}
        # {'race': {' Asian-Pac-Islander', ' Other', ' White', ' Amer-Indian-Eskimo', ' Black'}}
        # {'sex': {' Male', ' Female'}}
        # {'native-country': {' France', ' Japan', ' Iran', ' Taiwan', ' Hungary', ' Trinadad&Tobago',
        #                     ' Holand-Netherlands', ' Honduras', ' Outlying-US(Guam-USVI-etc)', ' Vietnam', ' Canada',
        #                     ' Italy', ' South', ' Jamaica', ' Mexico', ' Philippines', ' Ecuador', ' Greece',
        #                     ' Nicaragua', ' Portugal', ' Columbia', ' Scotland', ' Yugoslavia', ' Dominican-Republic',
        #                     ' England', ' Guatemala', ' United-States', ' Peru', ' Laos', ' Germany', ' Hong',
        #                     ' El-Salvador', ' Ireland', ' Haiti', ' India', ' Poland', ' Cambodia', ' Puerto-Rico',
        #                     ' Thailand', ' Cuba', ' China'}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge',1),
                       'sex': ('fix', 1),
                       'race': ('fix', 1),
                       'native-country': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## COMPAS data set
    elif dataset['name'] == 'compas-scores-two-years':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., LIMIT_BAL) and features with set {} values (e.g., SEX) are discrete

        # {'age': [18, 96]}
        # {'priors_count': [0, 38]}
        # {'days_b_screening_arrest': [0, 1057]}
        # {'length_of_stay': [0, 799]}
        # {'is_recid': {0, 1}}
        # {'age_cat': {'25 - 45', 'Greater than 45', 'Less than 25'}}
        # {'c_charge_degree': {'F', 'M'}}
        # {'is_violent_recid': {0, 1}}
        # {'two_year_recid': {0, 1}}
        # {'sex': {0, 1}}
        # {'race': {'African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other'}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge',1),
                       'sex': ('fix', 1),
                       'race': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Credit card default data set
    elif dataset['name'] == 'credit-card-default':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., LIMIT_BAL) and features with set {} values (e.g., SEX) are discrete

        # {'LIMIT_BAL': [10000, 1000000]}
        # {'AGE': [21, 79]}
        # {'BILL_AMT1': [-165580, 964511]}
        # {'BILL_AMT2': [-69777, 983931]}
        # {'BILL_AMT3': [-157264, 1664089]}
        # {'BILL_AMT4': [-170000, 891586]}
        # {'BILL_AMT5': [-81334, 927171]}
        # {'BILL_AMT6': [-339603, 961664]}
        # {'PAY_AMT1': [0, 873552]}
        # {'PAY_AMT2': [0, 1684259]}
        # {'PAY_AMT3': [0, 896040]}
        # {'PAY_AMT4': [0, 621000]}
        # {'PAY_AMT5': [0, 426529]}
        # {'PAY_AMT6': [0, 528666]}
        # {'SEX': {1, 2}}
        # {'EDUCATION': {0, 1, 2, 3, 4, 5, 6}}
        # {'MARRIAGE': {0, 1, 2, 3}}
        # {'PAY_0': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_2': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_3': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_4': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_5': {0, 2, 3, 4, 5, 6, 7, 8, -1, -2}}
        # {'PAY_6': {0, 2, 3, 4, 5, 6, 7, 8, -1, -2}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'AGE': ('ge', 1),
                       'SEX': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## HELOC data set
    elif dataset['name'] == 'heloc':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., ExternalRiskEstimate)

        # {'ExternalRiskEstimate': [33, 94]},
        # {'MSinceOldestTradeOpen': [2, 803]},
        # {'MSinceMostRecentTradeOpen': [0, 227]},
        # {'AverageMInFile': [4, 322]},
        # {'NumSatisfactoryTrades': [0, 79]},
        # {'NumTrades60Ever2DerogPubRec': [0, 19]},
        # {'NumTrades90Ever2DerogPubRec': [0, 19]},
        # {'PercentTradesNeverDelq': [0, 100]},
        # {'MSinceMostRecentDelq': [0, 83]},
        # {'MaxDelq2PublicRecLast12M': [0, 9]},
        # {'MaxDelqEver': [2, 8]},
        # {'NumTotalTrades': [0, 104]},
        # {'NumTradesOpeninLast12M': [0, 19]},
        # {'PercentInstallTrades': [0, 100]},
        # {'MSinceMostRecentInqexcl7days': [0, 24]},
        # {'NumInqLast6M': [0, 66]},
        # {'NumInqLast6Mexcl7days': [0, 66]},
        # {'NetFractionRevolvingBurden': [0, 232]},
        # {'NetFractionInstallBurden': [0, 471]},
        # {'NumRevolvingTradesWBalance': [0, 32]},
        # {'NumInstallTradesWBalance': [1, 23]},
        # {'NumBank2NatlTradesWHighUtilization': [0, 18]},
        # {'PercentTradesWBalance': [0, 100]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Wine data set
    elif dataset['name'] == 'wine':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., alcohol)

        # {'alcohol': [11.03, 14.83]},
        # {'malic_acid': [0.74, 5.8]},
        # {'ash': [1.36, 3.23]},
        # {'alcalinity_of_ash': [10.6, 30.0]},
        # {'magnesium': [70.0, 162.0]},
        # {'total_phenols': [0.98, 3.88]},
        # {'flavanoids': [0.34, 5.08]},
        # {'nonflavanoid_phenols': [0.13, 0.66]},
        # {'proanthocyanins': [0.41, 3.58]},
        # {'color_intensity': [1.28, 13.0]},
        # {'hue': [0.48, 1.71]},
        # {'od280/od315_of_diluted_wines': [1.27, 4.0]},
        # {'proline': [278.0, 1680.0]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Heart disease data set
    elif dataset['name'] == 'heart-disease':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age) and features with set {} values (e.g., sex) are discrete

        # {'age': [29, 77]},
        # {'trestbps': [94, 200]},
        # {'chol': [126, 564]},
        # {'thalach': [71, 202]},
        # {'oldpeak': [0.0, 6.2]},
        # {'sex': {0, 1}},
        # {'cp': {1, 2, 3, 4}},
        # {'fbs': {0, 1}},
        # {'restecg': {0, 1, 2}},
        # {'exang': {0, 1}},
        # {'slope': {1, 2, 3}},
        # {'ca': {0.0, 1.0, 2.0, 3.0}},
        # {'thal': {3.0, 6.0, 7.0}}

        print('\n')
        print('---- user-specified constraints -----')
        constraints = {'age': ('ge', 1),
                       'sex': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

        ##IBA Seminar constrains



    elif dataset['name'] == 'IBA_seminar_dataset':
        print("hier")
        print(x_org)
        print("x org[1]")
        print(x_org[1])
        #[94, 200]
        constraints = {'Age': ('fix', 10),  # kann nur älter werden
                       'Occupation': ('fix', 2),  # Beruf sehr schwer zu ändern
                       'Annual_Income': ([0, x_org[1]*1.5],2), # Einkommen mittelschwer zu ändern
                       'Monthly_Inhand_Salary': ([0, x_org[2]*1.5],2),  # Einkommen mittelschwer zu ändern
                       #'Num_Bank_Accounts': ('fix', 1),  # kann leicht geändert werden
                       #'Num_Credit_Card': ('fix', 1),  # kann leicht geändert werden
                       'Interest_Rate': ('fix', 10),  # festgelegt, kann nicht geändert werden
                       #'Num_of_Loan': ('fix', 10),  # Vergangenheit -- kann nicht geändert werden
                       #'Type_of_Loan': ('fix', 5),  # nur sehr schwer zu ändern (interpretierbarkeit)
                       #'Delay_from_due_date': ('fix', 4),  # Vergangenheit -- kann nicht geändert werden
                       #'Num_of_Delayed_Payment': ('fix', 10),  # Vergangenheit -- kann nicht geändert werden
                       #'Changed_Credit_Limit': ('fix', 5),  # Balance mittelschwer zu ändern
                       #'Num_Credit_Inquiries': ('fix', 10),  # Vergangenheit -- kann nicht geändert werden
                       #'Credit_Mix': ('fix', 3),  # kann bei aktuellem angepasst werden (interpretierbarkeit)
                       'Outstanding_Debt': ([0, x_org[11]*1.5],2), # ausstehende Schulden nur mittelschwer zu ändern
                       #'Credit_Utilization_Ratio': ('fix', 3),  # nur mittelschwer zu ändern
                       'Credit_History_Age': ('fix', 10),  # Vergangenheit -- kann nicht geändert werden
                       #'Payment_of_Min_Amount': ('fix', 5),  # nur sehr schwer zu ändern
                       #'Total_EMI_per_month': ('fix', 5),  # nur sehr schwer zu ändern
                       'Amount_invested_monthly': ([0, x_org[14]*5],2)  # nur sehr schwer zu ändern
                       #'Payment_Behaviour': ('fix', 5),  # nur sehr schwer zu ändern
                       #'Monthly_Balance': ('fix', 3),  # Balance mittelschwer zu ändern
                       # Month, SSN, ID, Customer_ID: komplett raus, weil nicht interpretierbar
                       }

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

        #### END ###############################################IBA SEMINAR DATA PREPARATION########################################

        '''
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., CRIM) and features with set {} values (e.g., CHAS) are discrete

        # {'MedInc': [0.4999, 15.0001]},
        # {'HouseAge': [1.0, 52.0]},
        # {'AveRooms': [0.8461538461538461, 141.9090909090909]},
        # {'AveBedrms': [0.3333333333333333, 34.06666666666667]},
        # {'Population': [3.0, 35682.0]},
        # {'AveOccup': [0.6923076923076923, 1243.3333333333333]},
        # {'Latitude': [32.54, 41.95]},
        # {'Longitude': [-124.35, -114.31]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')
        '''

    ## Iris data set
    elif dataset['name'] == 'iris':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., sepal length (cm))

        # {'sepal length (cm)': [4.3, 7.9]},
        # {'sepal width (cm)': [2.0, 4.4]},
        # {'petal length (cm)': [1.0, 6.9]},
        # {'petal width (cm)': [0.1, 2.5]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Diabetes data set
    elif dataset['name'] == 'diabetes':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age)

        # {'age': [-0.107225631607358, 0.110726675453815]},
        # {'sex': [-0.044641636506989, 0.0506801187398187]},
        # {'bmi': [-0.0902752958985185, 0.17055522598066]},
        # {'bp': [-0.112399602060758, 0.132044217194516]},
        # {'s1': [-0.126780669916514, 0.153913713156516]},
        # {'s2': [-0.115613065979398, 0.198787989657293]},
        # {'s3': [-0.10230705051742, 0.181179060397284]},
        # {'s4': [-0.076394503750001, 0.185234443260194]},
        # {'s5': [-0.126097385560409, 0.133598980013008]},
        # {'s6': [-0.137767225690012, 0.135611830689079]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge', 1),
                       'sex': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## California Housing data set
    elif dataset['name'] == 'california-housing':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., CRIM) and features with set {} values (e.g., CHAS) are discrete

        # {'MedInc': [0.4999, 15.0001]},
        # {'HouseAge': [1.0, 52.0]},
        # {'AveRooms': [0.8461538461538461, 141.9090909090909]},
        # {'AveBedrms': [0.3333333333333333, 34.06666666666667]},
        # {'Population': [3.0, 35682.0]},
        # {'AveOccup': [0.6923076923076923, 1243.3333333333333]},
        # {'Latitude': [32.54, 41.95]},
        # {'Longitude': [-124.35, -114.31]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    print('\n')
    print('N.B. preferences are taken into account when ACTIONABILITY=True!')
    print('\n')

    preferences = {'constraint': constraint,
                   'importance': importance}

    return preferences