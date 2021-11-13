from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
import numpy as np
import pandas as pd

class ada_boost:

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alpha = np.zeros(n_estimators)
        self.h = []

    def boost(self, X_train, y_train):
        adult_sum = X_train.shape[0]
        D = np.ones(adult_sum)/adult_sum

        for i in range(self.n_estimators):

            dtc = tree.DecisionTreeClassifier()
            dtc = dtc.fit(X_train, y_train, D)
            pred = dtc.predict(X_train)

            error = np.average(pred != y_train, weights=D)
            if error>0.5:
                break

            self.h.append(dtc)
            self.alpha[i] = 0.5 * np.log((1 - error) / error)
            D = D*np.exp(-1*pred*y_train*self.alpha[i])
            D = D/np.sum(D)

    def predict(self, X_test):
        clf_num = len(self.h)
        pred_matrix = np.zeros((clf_num, X_test.shape[0]))
        result = np.zeros(X_test.shape[0])
        prob = np.zeros(X_test.shape[0])

        for i in range(clf_num):
            pred_matrix[i] = self.h[i].predict(X_test)*self.alpha[i]
            prob += pred_matrix[i]
        
        for i in range(X_test.shape[0]):
            if X_test[i][9]!=0:
                prob[i] += (X_test[i][9]-4000)/6000
            if prob[i] >= 0:
                result[i] = 1
            else:
                result[i] = -1
        
        return result, prob

def cross_validation(X_train, y_train):
    skf = StratifiedKFold()
    opt_T = -1
    max_auc = 0

    for t in range(1,51):
        temp_auc = 0
        for train_idx, cv_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_cv, y_cv = X_train[cv_idx], y_train[cv_idx]
            adaboost = ada_boost(t)
            adaboost.boost(X_tr, y_tr)
            cv_prob = adaboost.predict(X_cv)[1]
            temp_auc += metrics.roc_auc_score(y_cv, cv_prob)
        temp_auc /= 5
        if temp_auc > max_auc:
            opt_T = t
            max_auc = temp_auc
        print(t, temp_auc)
    
    return opt_T



def clean_method(dataset, check_student=0, check_retired=0, never_worked = 1):
    
    for i in range(dataset.shape[0]):
        if check_student:
            if (dataset['Age'][i]<24 and dataset['Occupation'][i]==' ?'):
                dataset['Workclass']=='Never-worked'
                dataset['Occupation']='Student'
        
        if never_worked:
            if (dataset['Age'][i]>24 and dataset['Workclass'][i]==' Never-worked'):
                dataset['Occupation']='None'
        
        if check_retired:
            if (dataset['Age'][i]>60 and dataset['Workclass'][i]==' ?'):
                dataset['Workclass']=='Retired'
                dataset['Occupation']='Retired'
    
    return dataset


def data_clean(dirty_dataset, dcflag):
    if dcflag == 0:
        return dirty_dataset
    elif dcflag == 1:
        return clean_method(dirty_dataset, 1, 1)


def data_transform(org_dataset, dcflag=0):
    columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus',
        'Occupation','Relationship','Race','Sex','CapitalGain',
        'CapitalLoss','HoursPerWeek','Country','Income']
    
    if org_dataset == 'adult.data':
        rd_dataset = pd.read_csv(org_dataset, names=columns)
    elif org_dataset == 'adult.test':
        rd_dataset = pd.read_csv(org_dataset, names=columns, skiprows=1)
    
    rd_dataset.drop('fnlgwt', axis=1, inplace=True)
    rd_dataset = data_clean(rd_dataset, dcflag)
    
    args = list()
    for i in rd_dataset.columns:
        rd_dataset[i].replace('?', 'null', inplace=True)
        if rd_dataset[i].dtype != 'int64':
            rd_dataset[i] = rd_dataset[i].apply(lambda val: val.replace(" ", ""))
            rd_dataset[i] = rd_dataset[i].apply(lambda val: val.replace(".", ""))
            args.append((i, LabelEncoder()))

    data_mapper = DataFrameMapper(args, df_out=True, default=None)
    dataset = data_mapper.fit_transform(rd_dataset.copy())
    X_columns = list(rd_dataset.columns.copy())
    X_columns.remove('Income')

    X,y = dataset[X_columns].values, dataset['Income'].values

    return X,y

if __name__ == "__main__":
    X_train, y_train = data_transform('adult.data', 1)
    X_test, y_test = data_transform('adult.test', 1)

    y_train = (y_train - 0.5)*2
    y_test = (y_test - 0.5)*2

    #T = cross_validation(X_train, y_train)
    adaboost = ada_boost(500)
    adaboost.boost(X_train, y_train)
    yPred, yProb = adaboost.predict(X_test)

    print(metrics.classification_report(y_true=y_test, y_pred=yPred))
    print("AdaBoost AUC: ", metrics.roc_auc_score(y_test, yProb))