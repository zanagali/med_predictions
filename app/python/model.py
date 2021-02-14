#imports
#-----general libraries-----
import pandas as pd
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----sklearn-libraries-----
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
#-----others-----------------   

from xgboost import XGBClassifier
import shap
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from statistics import mean
#Importing data, password-protected
import pandas as pd
import xlwings as xw
import sys
from sklearn.metrics import roc_curve, auc

sys.path.insert(1, 'C:/Users/Zana/anaconda3/envs/PIPRA/Work_pipra/')
from model_helper_functions import modelHelper
from app.python.data import Data
from data_cleaning import Ana



class Model():
    
    def forecast(self,df_transUC):
        mh = modelHelper()
        #best_model from grid_search and randomSearch
        best_model = Pipeline(steps=[('preprocessor',ColumnTransformer(remainder='passthrough',
                        transformers=[('num',Pipeline(steps=[('scaler',StandardScaler())]),['age', 'weight (kg)',
                        'height (m)','BMI_value'])])),('clf',RandomForestClassifier(max_depth=26, n_estimators=451,
                        random_state=19))])
        
     
        #Data Split
        y = df_transUC.iloc[:,0:1]
        X = df_transUC.iloc[:,1:]
        X_trainUC, X_testUC, y_trainUC, y_testUC = train_test_split(X, y, test_size=0.2, random_state=19)
        #predict
        y_pred = mh.applyBestEstimator(best_model,X_trainUC,y_trainUC, X_testUC,y_testUC)
        return df_transUC,X_trainUC,X_testUC,y_trainUC,y_testUC,y_pred


    def predictProba(self,data):
        dtf = data
        #dtf = pd.DataFrame(dtf)
        y = dtf.iloc[:,0:1]
        X = dtf.iloc[:,1:]
        X_trainUC, X_testUC, y_trainUC, y_testUC = train_test_split(X, y, test_size=0.2, random_state=19)
        y_Trvalues = y_trainUC.values
        X_Trvalues = X_trainUC.values
        X_Tevalues = X_testUC.values

        best_model = Pipeline(steps=[('preprocessor',ColumnTransformer(remainder='passthrough',
                        transformers=[('num',Pipeline(steps=[('scaler',StandardScaler())]),['age', 'weight (kg)',
                        'height (m)','BMI_value'])])),('clf',RandomForestClassifier(max_depth=26,n_estimators=451,
                        random_state=19))])
        best_model.fit(X_trainUC,y_trainUC)
        y_score = best_model.predict_proba(X_testUC)[:, 1]
        #fpr, tpr, thresholds = roc_curve(y_testUC, y_score)
        return y_score,y_testUC.values.ravel()



