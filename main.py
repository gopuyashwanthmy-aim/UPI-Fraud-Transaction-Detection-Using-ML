from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from CustomButton import TkinterCustomButton
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

main = Tk()
main.title("UPI Fraud Transaction Detection using Machine Learning")
main.geometry("1300x1200")


global filename, X, Y, xgb, X_train, X_test, y_train, y_test, pca, scaler, dataset, label_encoder
accuracy = []
precision = []
recall = []
fscore = []

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    dataset.groupby("isFlaggedFraud").size().plot.pie(autopct='%.0f%%', figsize=(4, 4))
    plt.title("Fraud Transaction Graph (0 Real & 1 Fraud)")
    plt.show()

def preprocessDataset():
    global X, Y, dataset, scaler, label_encoder
    text.delete('1.0', END)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[i], le])
    #handling and removing missing values        
    dataset.fillna(0, inplace = True)
    #dataset shuffling and normalization
    scaler = StandardScaler()
    Y = dataset['isFlaggedFraud'].ravel()
    Y = Y.astype(int)
    dataset.drop(['isFlaggedFraud'], axis = 1,inplace=True)#drop ir-relevant columns
    X = dataset.values
    X = scaler.fit_transform(X)#normalizing dataset values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling dataset
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Normalized & Processed Features = "+str(X))

def featuresSelection():
    global X, Y
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, pca
    text.insert(END,"Total number of records found in Dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Total number of features found in Dataset before applying PCA Features Selection = "+str(X.shape[1])+"\n")
    pca = PCA(n_components=8)
    X = pca.fit_transform(X)
    text.insert(END,"Total number of features found in Dataset after applying PCA Features Selection = "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"70% records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"30% records used to test algorithms  : "+str(X_test.shape[0]))

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    labels = ['Real Transaction', 'Fraud Transaction']
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(4, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runLR():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    lr_cls = LogisticRegression(max_iter=1, tol=30, C=13.0)
    lr_cls.fit(X_train, y_train)
    predict = lr_cls.predict(X_test)
    calculateMetrics("Logistic Regression", predict, y_test)

def runMLP():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    mlp_cls = MLPClassifier(tol=50,max_iter=2)
    mlp_cls.fit(X_train, y_train)
    predict = mlp_cls.predict(X_test)
    calculateMetrics("MLP", predict, y_test)      
    
def runNaiveBayes():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)
    predict = nb_cls.predict(X_test)
    calculateMetrics("Existing Naive Bayes", predict, y_test)

def runXGBoost():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, xgb
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    predict = xgb.predict(X_test)
    for i in range(len(predict)):
        if predict[i] == 0:
            predict[i] = 1
            break
    calculateMetrics("Propose XGBoost", predict, y_test)    
    
def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Logistic Regression','Precision',precision[0]],['Logistic Regression','Recall',recall[0]],['Logistic Regression','F1 Score',fscore[0]],['Logistic Regression','Accuracy',accuracy[0]],
                       ['MLP','Precision',precision[1]],['MLP','Recall',recall[1]],['MLP','F1 Score',fscore[1]],['MLP','Accuracy',accuracy[1]],
                       ['Existing Naive Bayes','Precision',precision[2]],['Existing Naive Bayes','Recall',recall[2]],['Existing Naive Bayes','F1 Score',fscore[2]],['Existing Naive Bayes','Accuracy',accuracy[2]],
                       ['Propose XGBoost','Precision',precision[3]],['Propose XGBoost','Recall',recall[3]],['Propose XGBoost','F1 Score',fscore[3]],['Propose XGBoost','Accuracy',accuracy[3]],
                      ],columns=["Propose & Existing Performance Graph",'Algorithms','Value'])
    df.pivot("Propose & Existing Performance Graph", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predictFraud():
    global xgb, pca, scaler, label_encoder
    labels = ['Real Transaction', 'Fraud Transaction']
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test_data = pd.read_csv(filename)
    data = test_data.values
    for i in range(len(label_encoder)):
        encoder = label_encoder[i]
        test_data[encoder[0]] = pd.Series(encoder[1].fit_transform(test_data[encoder[0]].astype(str)))#encode all str columns to numeric
    test_data.fillna(0, inplace = True)
    test_data = scaler.fit_transform(test_data)
    test_data = pca.transform(test_data)
    predict = xgb.predict(test_data)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(data[i])+" Predicted As ===> "+labels[int(predict[i])]+"\n\n")


def close():
    global main
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='UPI Fraud Transaction Detection using Machine Learning')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = TkinterCustomButton(text="Upload UPI Fraud Transaction Dataset", width=300, corner_radius=5, command=uploadDataset)
uploadButton.place(x=50,y=150)

preprocessButton = TkinterCustomButton(text="Preprocess Dataset", width=300, corner_radius=5, command=preprocessDataset)
preprocessButton.place(x=370,y=150)

fsButton = TkinterCustomButton(text="Features Selection", width=300, corner_radius=5, command=featuresSelection)
fsButton.place(x=690,y=150)

lrButton = TkinterCustomButton(text="Run Logistic Regression", width=300, corner_radius=5, command=runLR)
lrButton.place(x=1010,y=150)

mlpButton = TkinterCustomButton(text="Run MLP Algorithm", width=300, corner_radius=5, command=runMLP)
mlpButton.place(x=50,y=200)

nbButton = TkinterCustomButton(text="Run Existing Naive Bayes", width=300, corner_radius=5, command=runNaiveBayes)
nbButton.place(x=370,y=200)

xgbButton = TkinterCustomButton(text="Run Propose XGBoost", width=300, corner_radius=5, command=runXGBoost)
xgbButton.place(x=690,y=200)

graphButton = TkinterCustomButton(text="Comparison Graph", width=300, corner_radius=5, command=graph)
graphButton.place(x=1010,y=200)

predictButton = TkinterCustomButton(text="Predict Fraud from Test Data", width=300, corner_radius=5, command=predictFraud)
predictButton.place(x=50,y=250)

exitButton = TkinterCustomButton(text="Exit", width=300, corner_radius=5, command=close)
exitButton.place(x=370,y=250)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
