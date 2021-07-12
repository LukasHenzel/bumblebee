# C19 - Der Chatbot zum Thema Corona
Unser Projekt widmet sich der Erstellung eines Chatbots mit Hilfe von Lernalgorithmen rund um das Thema Corona. Hierzu haben wir im Laufe dieses Projektes den C19 Chatbot entwickelt. Der User sollte in der Lage sein dem Chatbot Fragen zum Thema Corona zu stellen und einen Selbsttest durchführen zu können, welcher dem Nutzer einen Indikator  bezüglich einer möglichen Erkrankung gibt. 
Das Datenset zu den Symptomen einer Corona-Erkranung im Frühstadium ist auf Kaggle unter folgender URL zu finden: https://www.kaggle.com/martuza/early-stage-symptoms-of-covid19-patients .   
Das Datenset, welches für den Chatbot verwendet wurde, ist in dem Github-Ordner Code unter dem Namen intents.json zu finden.

Die Gruppe bumblebee besteht aus 2 Personen: 
Anne-Sophie Amberger 7053036 und
Lukas Henzel 2539341

Um den Quellcode auszuführen wurde eine spezielle Ananconda Umgebung(Diese Datei befindet sich hier in Github unter dem Ordner Umgebung) verwendet. Um die gleiche Entwicklungsumgebung zu erzeugen, kann eine YAML-Datei importiert werden. Dies geht mit dem folgenden Befehl: conda env create -f environment.yml                 
Hiernach kann die Umgebung mit folgendem Befehl aktiviert werden: conda activate tensor_keras                                           
Mit dem Befehl conda env list kann die korrekte Installation überprüft werden.

Unter folgendem Link findet man eine Anleitung zum Import von Umgebungen: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

Im Ordner Code sind alle Dateien, welche für die Ausführung benötigt werden, vorhanden. Nun kann die Datei chatbot_hanover.py ausgeführt werden.
Die Datei chatbot_ki_hanover.py erstellt das Modell chatbot_hanover_model.h5, welches der Chatbot benutzt. Mit der Datei symptoms_ai.py kann das Modell für den Selbsttest(self_pred.joblib) erstellt werden. 
