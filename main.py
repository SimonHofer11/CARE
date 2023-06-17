#Importing packages
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from user_preferences import userPreferences
from care_explainer import CAREExplainer
import pickle


def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'

    # defining the list of data sets
    datsets_list = {
        'IBA_seminar_dataset': ("preprocessed_data.csv", PrepareIBA_dataset, 'classification'),
    }

    # defining the list of black-boxes
    #man kann theoretisch auch GradientboostingClassifier für unseren Fall anwenden,
    #für MLPClassifier müsste man zusätzlich noch Implementierungsschritte codieren
    blackbox_list = {
        'rf-c': RandomForestClassifier
        #'nn-c': MLPClassifier,
        #'gb-c': GradientBoostingClassifier
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=45)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn-c':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # instance to explain
            cf_list = {}
            #Loop for running through several facts
            for i in range(47,63):
                x_ord = X_test[i]
                n_cf = 5

                # set user preferences || they are taken into account when ACTIONABILITY=True!
                user_preferences = userPreferences(dataset, x_ord)

                # explain instance x_ord using CARE
                output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                       SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True,
                                       user_preferences=user_preferences, cf_class='neighbor',
                                       probability_thresh=0.450
                                       , cf_quantile='neighbor', n_cf=n_cf)

                # Append the value of i to the data list
                print(output['x_cfs_highlight'])
                cf_list['test_' + str(i) + '_highlight'] = output['x_cfs_highlight']
                cf_list['test_' + str(i) + '_eval'] = output['x_cfs_eval']
                cf_list['test_' + str(i) + '_next'] = "--------------------NEXT COUNTERFACTUAL----------------------------"

            # create a binary pickle file with results
            with open('cf_seminar_results_file.pickle', 'wb') as f:
                pickle.dump(cf_list, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()









