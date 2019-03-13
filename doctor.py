from datetime import datetime
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer
import string
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
import xgboost
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, libsvm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import fbeta_score as fb, f1_score as f1, silhouette_score as shs, mean_squared_log_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import RFE, SelectKBest, chi2, SelectFromModel
import numpy as np
from sklearn.cluster import KMeans, hierarchical 
from sklearn.metrics.pairwise import euclidean_distances
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from textblob import TextBlob, Word
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from nltk.corpus import stopwords
from statsmodels.stats.outliers_influence import variance_inflation_factor
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer, MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split as tts, KFold
from keras.models import Sequential
from keras.layers import Dense
stop_words = stopwords.words('english')
ss = StandardScaler()
mabs = MaxAbsScaler()
pca = PCA(500)
le = LabelEncoder()
mms = MinMaxScaler()
imp = Imputer()
lemmatizer = WordNetLemmatizer()

df  = pd.read_excel('Doctor/train.xlsx', header = 0, index_col = None)

df_test = pd.read_excel('Doctor/test.xlsx', header = 0, index_col = None)



df.info()

exp = df.iloc[:, 1].apply(lambda x : x.split(' ')[0])

df_1 = df.drop('Experience', axis = 1)

df_2 = pd.concat([df_1, exp], 1)

for i in range(df_2.shape[0]):
    if type(df_2.iloc[i, 2]) != str:
        df_2.iloc[i, 2] = 'None'

place = df_2.loc[:, 'Place'].apply(lambda x : x.split(',')[-1])

df_3 = df_2.drop('Place', axis = 1)

df_4 = pd.concat([df_3, place], 1)

df_4.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])

df_final = df_4.drop(['Rating', 'Miscellaneous_Info'], axis = 1)

df_final.iloc[:, -2] =  df_final.iloc[:, -2].astype('int64')

df_final.iloc[:, -1] = le.fit_transform(df_final.iloc[:, -1])

df_final.iloc[:, 1] = le.fit_transform(df_final.iloc[:, 1])

numeric_features = df_final.dtypes[df_final.dtypes != 'object'].index
spearman = df_final[numeric_features].corr(method='spearman')
corr_with_target = spearman.ix[-3][1:-1]
corr_with_target = corr_with_target[abs(corr_with_target).argsort()[::-1]]
corr_with_target

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_final.values, i) for i in range(df_final.shape[1])]
vif["features"] = df_final.columns

X = df_final.loc[:, ['Qualification', 'Profile', 'Experience', 'Place']]
y = df_final.iloc[:, 2]

X = pd.get_dummies(X, columns = ['Qualification', 'Profile', 'Place'], drop_first = 1)

X = mms.fit_transform(X)

X = pca.fit_transform(X)

X = pd.DataFrame(X)


kf = KFold(n_splits = 7, shuffle = True, random_state = 1)
for i, j in kf.split(X):
    X_train, X_test = X.iloc[i], X.iloc[j] 
    y_train, y_test = y.iloc[i], y.iloc[j]

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = abs(lr.predict(X_test))

lr_error = np.sqrt(mean_squared_log_error(y_test, pred_lr))

rfr = RandomForestRegressor(random_state = 1, max_features = 'auto')
rfr.fit(X_train, y_train)

pred_rfr = abs(rfr.predict(X_test))

rfr_error = np.sqrt(mean_squared_log_error(y_test, pred_rfr))

gbr = GradientBoostingRegressor(random_state = 1)
gbr.fit(X_train, y_train)

pred_gbr = abs(gbr.predict(X_test))

gbr_error = np.sqrt(mean_squared_log_error(y_test, pred_gbr))

knr = KNeighborsRegressor(n_neighbors = 7, p = 3)
knr.fit(X_train, y_train)

pred_knr = abs(knr.predict(X_test))

knr_error = np.sqrt(mean_squared_log_error(y_test, pred_knr))




