from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV

def get_svm_model(parameters=None):
    svc = svm.SVC()  # Support Vector Classifier
    clf = GridSearchCV(svc, parameters)  # Grid search over `kernel` and `C`
    return clf