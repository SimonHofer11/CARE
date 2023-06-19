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

    if dataset['name'] == 'IBA_seminar_dataset':
        #für manche numerische Variablen haben wir implementiert,
        #dass Wertebereich des foils beispielsweise bei max. 50% höher als Ausgangswert sein soll->realistischere Änderungen

        constraints = {'Age': ('fix', 10),
                       'Occupation': ('fix', 2),
                       'Annual_Income': ([0, x_org[1]*1.5],2),
                       'Monthly_Inhand_Salary': ([0, x_org[2]*1.5],2),
                       #'Num_Bank_Accounts': ('fix', 1),
                       #'Num_Credit_Card': ('fix', 1),
                       'Interest_Rate': ('fix', 10),
                       #'Num_of_Loan': ('fix', 10),
                       #'Type_of_Loan': ('fix', 5),
                       #'Delay_from_due_date': ('fix', 4),
                       #'Num_of_Delayed_Payment': ('fix', 10),
                       #'Changed_Credit_Limit': ('fix', 5),
                       #'Num_Credit_Inquiries': ('fix', 10),
                       #'Credit_Mix': ('fix', 3),
                       'Outstanding_Debt': ([0, x_org[11]*1.5],2),
                       #'Credit_Utilization_Ratio': ('fix', 3),
                       'Credit_History_Age': ('fix', 10),
                       #'Payment_of_Min_Amount': ('fix', 5),
                       #'Total_EMI_per_month': ('fix', 5),
                       'Amount_invested_monthly': ([0, x_org[14]*5],2)
                       #'Payment_Behaviour': ('fix', 5),
                       #'Monthly_Balance': ('fix', 3),
                       }

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