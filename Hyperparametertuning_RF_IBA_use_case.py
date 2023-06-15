from prepare_datasets import PrepareIBA_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


####Parametertuning####
"""
df = pd.read_csv(preprocessed_data.csv',
    sep=',')
df.drop(['ID', 'Customer_ID', 'SSN'], axis=1, inplace=True)

## Handling missing values
df = df.dropna().reset_index(drop=True)
"""
x = 2
y = 2

df = PrepareIBA_dataset(x, y)
X, y = df['X_ord'], df['y']

rf = RandomForestClassifier()

# Definieren der Hyperparameter, die getestet werden sollen
param_grid = {
    'n_estimators': [100, 200, 300],  # Anzahl der Bäume im Wald
    'max_depth': [None, 5, 10],  # maximale Tiefe der Bäume
    'min_samples_split': [2, 5, 10],  # Mindestanzahl von Samples für einen Split
    'min_samples_leaf': [1, 2, 4]  # Mindestanzahl von Samples in einem Blatt
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisierung der GridSearchCV mit dem Random Forest-Klassifikator und dem Parameterraster
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Durchführen der Suche nach den besten Hyperparametern
grid_search.fit(X_train, y_train)

# Ausgabe der besten Hyperparameter-Kombination
print("Beste Hyperparameter-Kombination gefunden:")
print(grid_search.best_params_)

# Auswertung des Modells auf den Testdaten
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Genauigkeit des besten Modells auf den Testdaten:", accuracy)
