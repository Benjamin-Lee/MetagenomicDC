import numpy as np
import sys
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from tpot import TPOTClassifier

#parameters: sys.argv[1] = input dataset as matrix of k-mers
from pathlib import Path

# Loading dataset
nome_train = Path(sys.argv[1]).stem
print(nome_train)

def load_data(file):
    lista = []
    records = list(open(file, "r"))
    records = records[1:]
    for seq in records:
        elements = seq.split(",")
        level = elements[-1].split("\n")
        classe = level[0]
        lista.append(classe)

    lista = set(lista)
    classes = list(lista)
    X = []
    Y = []
    for seq in records:
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
    # Best pipeline: KNeighborsClassifier(input_matrix, n_neighbors=68, p=2, weights=distance)
    # logistic regression
    classifier = TPOTClassifier(random_state=42, verbosity=2, config_dict='TPOT light', generations=2)
    return classifier

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test):

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print(acc)
    return Y_pred, Y_test


if __name__ == "__main__":
    X_train, Y_train, nb_classes, input_length = load_data(sys.argv[1])
    model = create_model()
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train,
                                                        train_size=0.75, test_size=0.25)
    train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test)
