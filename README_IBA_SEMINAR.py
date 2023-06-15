'''
Read Me um Ausführung von CARE auf IBA Satensatz zu verstehen:
#1. in README.md Erklärung, um Python Environment aufzusetzen.
#2.wichtige Files:
-prepreprocessed_data.csv: aufbereiteter Datensatz (in Seminararbeit zu Erklärung, was geändert wurde)
-prepare_datasets.py: hier haben wir den Datensatz so aufbereitet, dass man CARE darauf anwenden kann
-get_to_know_data.py: unsere Datenaufbereitung (Erkentnisse davon sind dann u.a in prepare_datasets.py eingeflossen)
-user_preferences.py: unsere Restriktionen die in Optimierungsproblem miteingebunden werden
-Hyperparametertuning_RF_IBA_use_case.py: Hyperparametertuning des Implementierten Random Forests auf unseren Datensatz
-main.py: hier wird alles zusammengeführt für counterfactual Erzeugung

Sonstige bearbeitete Files:
-create_model.py: hier wurde der Random Forest (mit Parametertuningsettings) hinzugefügt
-care.py: hier wurde bestimmt, welche Zielklasse ien foil haben soll (Poor->Standard,Standard->Good,Good->Standard)


->Environment set up abschließen
->main ausführen um counterfactuals zu erzeugen
'''