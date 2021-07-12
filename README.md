# C19
Unser Projekt widmet sich der Erstellung eines Chatbots mit Hilfe von Lernalgorithmen rund um das Thema Corona. Hierzu haben wir im Laufe dieses Projektes den C19 Chatbot entwickelt. Der User sollte in der Lage sein dem Chatbot Fragen zum Thema Corona zu stellen und einen Selbsttest durchführen zu können, welcher dem Nutzer einen Indikator  bezüglich einer möglichen Erkrankunbg gibt.
Das Datenset zu den Symptomen einer Corona-Erkranung im Frühstadium ist unter folgender URL zu finden: https://www.kaggle.com/martuza/early-stage-symptoms-of-covid19-patients .   
Das Datenset, welches für den Chatbot verwendet wurde ist unter dem Ordner Code zu finden(intents.json).

Die Gruppe bumblebee besteht aus 2 Personen: 
Anne-Sophie Amberger 7053036
Lukas Henzel 2539341

Um den Quellcode auszuführen wurde eine spezielle Ananconda Umgebung(datei befindet sich hier in Github unter dem Ordner Umgebung) verwendet. Um die Gleiche Umgebung zu erzeugen kann eine YAML-Datei importiert werden. Dies geht mit dem folgenden Befehl: conda env create -f environment.yml                 
Hiernach kann die Umgebung mit folgendem Befehl aktiviert werden: conda activate tensor_keras                                           
Mit dem Befehl conda env list kann die korrekte Installation überprüft werden.

Unter folgendem Link findet man eine Anleitung zum Import von Umgebungen: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
