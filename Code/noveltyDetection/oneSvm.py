""
# oneSvm for bankruptcyPrediction
""
#author shuoshuofan
#email  shuoshuofan@gmail.com

import  sys
sys.path.append("..")
import  load_data
import  numpy as np
import  matplotlib.pyplot as plt
import  matplotlib.font_manager
#sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#get data
bank2data=load_data.load_2year()
X=bank2data[0]
y=bank2data[1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

clf=svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.1)

clf.fit(X_train)
y_pred_train=clf.predict(X_train)


#print(y_pred_train)
y_pred_test=clf.predict(X_test)
#print(y_pred_test)

index=0
for row in y_pred_train:
    if(row==1):
        y_pred_train[index]=0
    else:
        y_pred_train[index]=1
    index+=1

index=0
for row in y_pred_test:
    if(row==1):
        y_pred_test[index]=0
    else:
        y_pred_test[index]=1
    index+=1

print("train data")
print(classification_report(y_train,y_pred_train))
print()
print("test data")
print(classification_report(y_test,y_pred_test))
print()



