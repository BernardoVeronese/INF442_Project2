import numpy as np
import matplotlib.pyplot as plt
from src.features.build_features import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC, LinearSVC

# Hyperparameters
p=13
q=2

# Data processing
data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
data_file = "EUKSIG_13.red.txt"
seq, cleav = read_sequence(data_path + data_file)
cleavpos = return_cleavpos(cleav)
alphabet = return_alphabet(seq)
d = dict_from_alphabet(alphabet)

sequence,labels=get_encoded_features2(data_path, p, q)
N=len(sequence)

################################################Similarity kernel

def K1(U,V):
    #U and V are encoded as 0s and 1s
    return (np.dot(U,V.T))

def K3(U,V) :
    #Uand V are sequences encoded by the number corresponding to the letter with d.
    n_samples_1=U.shape[0]
    n_samples_2=V.shape[0]

    W=np.eye(n_samples_1,n_samples_2)
    
    for k in range(n_samples_1) :
        for j in range(n_samples_2):
            count = 0
            for i in range(p+q) :
              if (U[k][i]==V[j][i]) :
                  count+=1
            W[k][j]=count
    return(W)           

#hyperparameters : substitution matrix and bandwidth
path="C:/Users/antoi/OneDrive/Bureau/Polytechnique-2A/P3/INF 422/PI/INF442_Project2-master/src/data/Substitution matrices/IDENTITY" #change it to your convenience to choose one of the matrices"
M=get_similarity_matrix(path,d)
print(M)
gamma=0.1

#Score
def s(a,b) :
  sum=0
  n=p+q
  for i in range(n) :
      sum+=M[int(a[i])][int(b[i])]
  return(sum)

def K2(U,V) :
    n_samples_1=U.shape[0]

    n_samples_2=V.shape[0]
    W=np.eye(n_samples_1,n_samples_2)
    
    for k in range(n_samples_1) :
        for j in range(n_samples_2):
            #W[k][j]=exp(-gamma*(s(U[k],V[j])))
            W[k][j]=s(U[k],V[j])
    return(W)

###Prediction with a given kernel

    
def PredictionCustomKernel(train_d,train_l,test_d,test_l):
    #trains a SVM with train_data labelled with train_labels, tests on test_data and computes accuracy
    #fitting
    print("fitting...")
    clf=svm.SVC(kernel=K2)#change to your convenience
    clf.fit(train_d,train_l)
    print("OK")

    #computing accuracy
    print("predicting...")
    TP=0
    FP=0
    FN=0
    TN=0
    cont=0
    for sequence in test_d :
      predict=clf.predict(sequence.reshape(1,-1))[0]
      if (predict==test_l[cont]) and (test_l[cont]==1) :
          TP+=1
          print("true positive")
      elif (predict==test_l[cont]) and (test_l[cont]!=1):
          TN+=1
      elif (predict!=test_l[cont]) and (test_l[cont]!=1):
          FP+=1
          print("false positive")
      elif (predict!=test_l[cont]) and (test_l[cont]==1):
          FN+=1
          print("false negative")
      cont+=1
    print(p)
    print(q)
    F1=(float(TP)/float(TP+FN+FP))
    print("F1 score computed :"+str(F1))
    balanced_accuracy=0.5*((float(TP)/float(TP+FP))+(float(TN)/float(TN+FN)))
    print("balanced_accuracy computed :"+str(balanced_accuracy))    

if __name__ == "__main__":


    """Csearch = np.logspace(-3, -1, num=5, base=10.0)
    print(Csearch)
    Csearch_dict = {'C': np.logspace(-1, 1, base=10)}"""
    help(seq_list_encoding)
    #clf = GridSearchCV(lsvc, param_grid=Csearch, refit=True)
    #clf.fit(X, Y)
    #print(clf.cv_results_)
    for iter in range(max_iter):
        p = np.random.randint(1, 15)
        q = np.random.randint(1, 15)
        X, Y = seq_list_encoding(seq, p + q, cleavpos, alphabet)
        lsvc = LinearSVC(C=0.01, class_weight='balanced', max_iter=5000)
        csv_score = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='accuracy')
        csv_score_balanced = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='balanced_accuracy')
        csv_score_roc = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='roc_auc')
        #print("(p, q) = (", p, q, ") Score (method = accuracy) : {:.4f}".format(np.mean(csv_score)))
        print("(p, q) = (", p, q, ") Score (method = balanced accuracy) : {:.4f}".format(np.mean(csv_score_balanced)))
        #print("(p, q) = (", p, q, ") Score (method = roc_auc) : {:.4f}".format(np.mean(csv_score_roc)))
