""
oneSvm for bankruptcyPrediction
""
#author shuoshuofan
#email  shuoshuofan@gmail.com


import  numpy as np
import  matplotlib.pyplot as plt
import  matplotlib.font_manager

from sklearn import svm
clf=svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.1)

