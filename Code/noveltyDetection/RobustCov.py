"""
Robust covriance
"""

#author shuoshuofan
#email  shuoshuofan@gmail.com

import  sys
sys.path.append("..")
import load_data
import numpy as np

#sklearn
from sklearn.covariance  import  EmpiricalCovariance,MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#get data
bank2data=load_data.load_2year()
X=bank2data[0]
y=bank2data[1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.31,random_state=0 )

#robust_cov=MinCovDet().fit(X_train)

emp_cov=EmpiricalCovariance().fit(X_train)
