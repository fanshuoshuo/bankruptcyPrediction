""
# oneSvm for bankruptcyPrediction
""
#author shuoshuofan
#email  shuoshuofan@gmail.com

import  sys
sys.path.append("..")
import  load_data
import  numpy as np
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
#split the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

clf=svm.OneClassSVM(nu=0.1,kernel="rbf")
#fit the data
clf.fit(X_train)
y_pred_train=clf.predict(X_train)
y_pred_test=clf.predict(X_test)
#replace(1,-1) to (0,-1)
y_pred_train=np.where(y_pred_train>0,0,1)
y_pred_test =np.where(y_pred_test >0,0,1)
#show the performance of the OneClassSVM
print("train data")
print(classification_report(y_train,y_pred_train))
print("f1_score is ",f1_score(y_train,y_pred_train))
print("roc_auc is ",roc_auc_score(y_train,y_pred_train))
print()
print("test data")
print(classification_report(y_test,y_pred_test))
print("f1_score is ",f1_score(y_test,y_pred_test))
print("roc_auc is ",roc_auc_score(y_test,y_pred_test))
print()



