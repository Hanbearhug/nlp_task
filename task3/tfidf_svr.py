import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, auc, precision_recall_curve
from sklearn.svm import LinearSVR
from utils import delete_stopwords, partition_df, read_stopwords
import os
print(os.listdir())

train = pd.read_table('cnews.train.txt', header=None, names=['type', 'content'])
test = pd.read_table('cnews.test.txt', header=None, names=['type', 'content'])

map_dict = dict(zip(train['type'].unique(), range(train['type'].nunique())))
train['type'] = train['type'].map(map_dict)
test['type'] = test['type'].map(map_dict)

stopwords = read_stopwords()
train = delete_stopwords(train, stopwords)
test = delete_stopwords(test, stopwords)

train = partition_df(train, type=False)
test = partition_df(test, type=False)

vect_word = TfidfVectorizer(max_features=20000, lowercase=False, analyzer='word', ngram_range=(1,3),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=20000, lowercase=False, analyzer='char', ngram_range=(3,6),dtype=np.float32)

print(train.loc[0,'content'])
# Word ngram vector
tr_vect = vect_word.fit_transform(train['content'])
ts_vect = vect_word.transform(test['content'])

# Character n gram vector
tr_vect_char = vect_char.fit_transform(train['content'])
ts_vect_char = vect_char.transform(test['content'])

X_train = hstack([tr_vect, tr_vect_char], 'csr')
X_test = hstack([ts_vect, ts_vect_char], 'csr')
y_train = train['type'].values
y_test = test['type'].values

NFold = 5
folds = StratifiedKFold(n_splits=NFold,shuffle=True,random_state=79)
oof_lgb_3 = np.zeros(len(train))
predictions_lgb_3 = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train,y_train)):
    print(f'fold{fold_} start ......')
    svr = LinearSVR(random_state=71, tol=1e-2, C=0.45,epsilon=0.2,max_iter=800)
    X_dtr,y_dtr = X_train[trn_idx], y_train[trn_idx]
    X_dval,y_dval = X_train[val_idx], y_train[val_idx]

    svr.fit(X_dtr, y_dtr)
    oof_lgb_3[val_idx] = svr.predict(X_train[val_idx])
    predictions_lgb_3 += svr.predict(X_test) / folds.n_splits
    print("CV score: {:<8.5f}".format(f1_score(y_train[val_idx], oof_lgb_3[val_idx], sample_weight='weighted')))
print("CV score: {:<8.5f}".format(f1_score(y_train, oof_lgb_3, sample_weight='weighted')))
print("CV score: {:<8.5f}".format(f1_score(y_test, predictions_lgb_3, sample_weight='weighted')))