import numpy as np
import sys
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#parameters: sys.argv[1] = input dataset as matrix of k-mers
from pathlib import Path
from datetime import datetime
from yaml import dump

from tqdm import tqdm

# Loading dataset
nome_train = Path(sys.argv[1]).stem
print(nome_train)

def load_data(file):
    lista = []
    records = list(open(file, "r"))
    records = records[1:]
    for seq in tqdm(records):
        elements = seq.split(",")
        level = elements[-1].split("\n")
        classe = level[0]
        lista.append(classe)

    lista = set(lista)
    classes = list(lista)
    X = []
    Y = []
    for seq in tqdm(records):
        elements = seq.split(",")
        X.append(elements[1:-1])
        level = elements[-1].split("\n")
        classe = level[0]
        Y.append(classes.index(classe))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=int)
    data_max = np.amax(X)
    X = X / data_max
    return X, Y, len(classes), len(X[0])


# Training
def create_model():
    classifier = linear_model.LogisticRegression(class_weight="balanced") # 88.3% with k=7
    return classifier

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test):
    train_start = datetime.now()
    model.fit(X_train, Y_train)
    train_duration_sec = (datetime.now() - train_start).seconds

    test_start = datetime.now()
    Y_pred = model.predict(X_test)
    test_duration_sec = (datetime.now() - test_start).seconds

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="weighted")
    recall = recall_score(Y_test, Y_pred, average="weighted")
    return dict(accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                train_duration_sec=train_duration_sec,
                test_duration_sec=test_duration_sec)


if __name__ == "__main__":
    X_train, Y_train, nb_classes, input_length = load_data(sys.argv[1])

    n_splits = 10

    results = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train, test in tqdm(kfold.split(X_train, Y_train), total=n_splits):
        model = create_model()
        results.append(train_and_evaluate_model(model, X_train[train], Y_train[train], X_train[test], Y_train[test]))

    dump(results, open(f"results/{nome_train}_results.yml", "w+"), default_flow_style=False)
