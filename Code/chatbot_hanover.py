# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:43:11 2021

@author: lukas
"""
from keras.models import load_model
model = load_model('chatbot_model_hanover.h5')
import json
import random
import pickle
import nltk
import numpy as np
import tkinter
from tkinter import *
import joblib
self_check_rf = joblib.load("./self_pred.joblib")

from HanTa import HanoverTagger as ht
hanover = ht.HanoverTagger('morphmodel_ger.pgz')

intents = json.loads(open('intents.json',encoding='utf-8').read(),strict=False)
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#Tokenizing und Lemmatization
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [(word.lower()) for word in sentence_words]
    sentence_words_lemma = []
    for word in sentence_words:
        #Hier könnte auch WordNetLemmatizer() aus der Bibliothek NLTK verwendet werden, falls es sich um einen englischen Text handelt
        lemma = [lemma for (word,lemma,pos) in hanover.tag_sent(word.split())]
        sentence_words_lemma.append(' '.join(lemma))
    return sentence_words_lemma

#Erstellung eines BoW mit 0 oder 1, je nachdem ob das Wort im BoW im Satz existiert
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # Wenn das Wort vorliegt wird eine 1 zugeordnet
                bag[i] = 1
                if show_details:
                    print ("Gefunden im BoW: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # Prediction des Modells mit einer Mindestwahrscheinlichkeit von 25% -> Kleine Werte werden direkt aussortiert
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Das Element mit der höchsten Wahrscheinlichkeit soll an der ersten Stelle stehen
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(return_list)  
        #Ermittelte Wahrscheinlichkeit ausgeben
    return return_list

# Nun muss noch die Antwort ausgewählt werden, hierzu wird diese aus der JSON-Datei ausgelesen
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    #Der Tag selbsttest ist eine besondere Kategorie und muss deshalb gesondert behandelt werden
    if tag =='selbsttest':
        result="Selbsttest wird gestartet..."
    else:
        for i in list_of_intents:
            if(i['tag']== tag):      
                result = random.choice(i['responses'])
                break
    return result

# Ausführung der Vorhersage und Auswahl der dazugehörigen Antwort
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Definition des Selbstchecker Formulars
def self_checker():
    self_form = Toplevel()
    def validate_int(entry):
        try:
            int(entry.get())
        except ValueError:
            messagebox.showerror(title="Fehler", message="Die eingegebene Nummer ist ungültig")
    
    sex = [("Männlich", 1),("Weiblich", 0)]
    label_0 = Label(self_form, text='Welches Geschlecht haben Sie?').pack()
    varsex = IntVar()
    for txt, val in sex:
        Radiobutton(self_form, text=txt, variable=varsex, value=val).pack()
    
    varage = StringVar()
    label_1 = Label(self_form, text="Alter").pack()
    entry_1 = Entry(self_form,textvariable=varage).pack()
    
    
    answer = [("Ja", 1),("Nein", 0)]
    
    label_2 = Label(self_form, text='Haben Sie Fieber?').pack()
    varfever = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varfever, value=val).pack()
        
    label_3 = Label(self_form, text='Haben Sie Husten?').pack()
    varcough = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varcough, value=val).pack()
        
    label_4 = Label(self_form, text='Haben Sie Schnupfen?').pack()
    varrunny_nose = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varrunny_nose, value=val).pack()
        
    label_5 = Label(self_form, text='Haben Sie Muskelkater?').pack()
    varmuscle_soreness = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varrunny_nose, value=val).pack()
        
    label_6 = Label(self_form, text='Haben Sie eine Pneumonie?').pack()
    varpneunomia = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varpneunomia, value=val).pack()
        
    label_7 = Label(self_form, text='Haben Sie eine Magen-Darm-Erkrankung?').pack()
    vardiarrhea = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=vardiarrhea, value=val).pack()
        
         
    label_8 = Label(self_form, text='Haben Sie eine Lungentzündung?').pack()
    varlung_infection = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varlung_infection, value=val).pack()
        
    label_9 = Label(self_form, text='Sind Sie in letzter Zeit gereist?').pack()
    vartravel_hist = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=vartravel_hist, value=val).pack()
        
    label_10 = Label(self_form, text='Haben Sie sich selbst isoliert?').pack()
    varself_isolation = IntVar()
    for txt, val in answer:
        Radiobutton(self_form, text=txt, variable=varself_isolation, value=val).pack()
    
    def get_data(varsex,varage,varfever,varcough,varrunny_nose,varmuscle_soreness,varpneunomia,vardiarrhea,varlung_infection,vartravel_hist,varself_isolation):
    
        rf_values_pred = ['gender','age','fever','cough','runny','muscle_soreness','pneumonia','diarrhea','lung_infection','travel_hist','isolation']
        rf_values_pred[0]=varsex.get()
        rf_values_pred[1]=varage.get()
        rf_values_pred[2]=varfever.get()
        rf_values_pred[3]=varcough.get()
        rf_values_pred[4]=varrunny_nose.get()
        rf_values_pred[5]=varmuscle_soreness.get()
        rf_values_pred[6]=varpneunomia.get()
        rf_values_pred[7]=vardiarrhea.get()
        rf_values_pred[8]=varlung_infection.get()
        rf_values_pred[9]=vartravel_hist.get()
        rf_values_pred[10]=varself_isolation.get()
        return rf_values_pred
    
    #Vorhersage des RandomForest und Ausgabes dessen Ergebisses   
    def predict_cov(varsex,varage,varfever,varcough,varrunny_nose,varmuscle_soreness,varpneunomia,vardiarrhea,varlung_infection,vartravel_hist,varself_isolation):
        global self_check_rf
        data=np.asarray(get_data(varsex,varage,varfever,varcough,varrunny_nose,varmuscle_soreness,varpneunomia,vardiarrhea,varlung_infection,vartravel_hist,varself_isolation)).reshape(1, -1)
        res = self_check_rf.predict(data)           
        if res[0] == 1:
            chat_log.config(state=NORMAL)
            chat_log.config(foreground="#442265", font=("Verdana", 12 ))
            chat_log.insert(END, "C19-Bot: Es besteht eine Wahrscheinlichkeit für eine Infektion.Bleiben Sie daher, bis Sie sich mit Ihrem Gesundheitsamt in Verbindung gesetzt haben und weitere Informationen erhalten zu Hause und isolieren Sie sich wenn es möglich ist."+ '\n\n')
            chat_log.config(state=DISABLED)
            chat_log.yview(END)
        elif res[0] == 0:
            chat_log.config(state=NORMAL)
            chat_log.config(foreground="#442265", font=("Verdana", 12 ))
            chat_log.insert(END, "C19-Bot: Es besteht keine hohe Wahrscheinlichkeit einer Infektion. Trotzdem sollten Sie ihre Symptome in den kommenden Tagen beobachten und bei Verschlechterung Rücksprache mit Ihrem Arzt halten oder sich bei ihrem zuständigen Gesundheitsamt melden." + '\n\n') 
            chat_log.config(state=DISABLED)
            chat_log.yview(END)
    
    def close():
        self_form.destroy()
        self_form.update()
    
    predict_button = Button(self_form, text = "Ergebnis", command=lambda:predict_cov(varsex,varage,varfever,varcough,varrunny_nose,varmuscle_soreness,varpneunomia,vardiarrhea,varlung_infection,vartravel_hist,varself_isolation)).pack()
    exit_button = Button(self_form, text = "Exit", command=close).pack()
    
    
     
#Gui erstellen mit tkinter

# Senden der Nachricht und desssen Interpretation mit Antwort
def send():
    msg = entry_box.get("1.0",'end-1c').strip()
    entry_box.delete("0.0",END)
    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "Sie: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        if res =="Selbsttest wird gestartet...":        
            self_checker()
        chat_log.insert(END, "C19-Bot: " + res + '\n\n')
        chat_log.config(state=DISABLED)
        chat_log.yview(END)
        
chatbot_window = Tk()
chatbot_window.title("C19-Bot")
chatbot_window.geometry("800x1000")
chatbot_window.resizable(width=FALSE, height=FALSE)
#Chatfenster kreieren
chat_log = Text(chatbot_window, bd=0, bg="white", height="16", width="100", font="Arial",)
chat_log.config(state=DISABLED)
scrollbar = Scrollbar(chatbot_window, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set
#Sendebutton erstellen
send_button = Button(chatbot_window, font=("Verdana",12,'bold'), text="Senden", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Hier steht der Text drin
entry_box = Text(chatbot_window, bd=0, bg="white",width="29", height="5", font="Arial")
scrollbar.place(x=752,y=12, height=772)
chat_log.place(x=12,y=12, height=772, width=740)
entry_box.place(x=256, y=802, height=180, width=530)
send_button.place(x=12, y=802, height=180)
chatbot_window.mainloop()
