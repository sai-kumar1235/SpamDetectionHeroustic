import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from genetic_selection import GeneticSelectionCV

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

global accuracy, precision, recall, fscore

accuracy = []
precision = []
recall = []
fscore = []

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens
'''
dataset = pd.read_csv("Dataset/spam_assassin.csv", nrows=2000)
dataset = dataset.dropna()
print(dataset.shape)
dataset = dataset.values

X = []
Y = []

for i in range(len(dataset)):
    mail = dataset[i,0]
    mail = mail.strip("\n").strip().lower()
    label = dataset[i,1]
    mail = cleanText(mail)#clean email
    X.append(mail)
    Y.append(label)
    print(str(i)+" "+str(len(mail))+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X", X)
np.save("model/Y", Y)
'''

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
print(X[12])
print(Y)

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=2000)
X = tfidf_vectorizer.fit_transform(X).toarray()
print(X)
print(X.shape)

sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(np.unique(Y, return_counts=True))


classifier = RandomForestClassifier()

def runGA():
    global X, Y, selector
    if os.path.exists("model/ga.npy"):
        selector = np.load("model/ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=10, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/ga", selector)
    X1 = X[:,selector] #extract selected features
    return X1, Y, selector

#PSO function
def f_per_particle(m, alpha):
    global X, Y, classifier
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def runPSO():
    global X, Y, pso
    if os.path.exists("model/pso.npy"):
        pso = np.load("model/pso.npy")
    else:
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
        dimensions = X.shape[1] # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
        cost, pso = optimizer.optimize(f, iters=35)#OPTIMIZING FEATURES
        np.save("model/pso", pso)
    X1 = X[:,pso==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
    return X1, Y

def calculateMetrics(algorithm, predict, y_test):
    label = ['Normal Event', 'Disaster Event']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runAlgorithm(X1, Y):
    X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2) #split dataset into train and test
    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)
    predict = nb_cls.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)    
    svm_cls = svm.SVC() #create SVM object
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)
    rf_cls = RandomForestClassifier() #create Random Forest object
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

if __name__=="__main__":
    X1, Y, selector = runGA()
    runAlgorithm(X1, Y)
    X1, Y = runPSO()
    runAlgorithm(X1, Y)
    print(accuracy)
    print(precision)
    print(recall)

