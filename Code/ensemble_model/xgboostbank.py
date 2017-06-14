"""
Random Forests to solve the bankruptcy prediction
"""
#author shuoshuofan
#email  shuoshuofan@gmail.com
import sys
sys.path.append("..")
import load_data
#sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
#xgboost
import xgboost as xgb
import numpy as np
#get data
bank2data=load_data.load_2year()

#bank2data=Imputer().transform(bank2data)
X=bank2data[0]
y=bank2data[1]
X=np.asanyarray(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.7,random_state=0)

tuned_parameters={"max_depth":[3,2,1],
                  "n_estimators":[15,20,25],}

scores=['recall_macro','roc_auc']

for score in scores:
    print("Tuning hyper-parameters for %s" %score)
    print()

    clf=GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'),tuned_parameters,cv=5,
                     scoring=score)
    clf.fit(X_train,y_train)

    print("Best parameters set found on devepment set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print(f1_score(y_true,y_pred))
    print(roc_auc_score(y_true,y_pred))
    print()



