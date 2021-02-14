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
sys.path.insert(1, 'C:/Users/Zana/anaconda3/envs/PIPRA/Work_pipra/')
from data_cleaning import Ana


class Data():
    def get_data(self):
        a = Ana()
        df = a.createdf()
        f_list = df.columns
        print(df.iloc[:,0:1])
        return df, f_list

