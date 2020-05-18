from src.models.Model import Model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from src.models.PositionScoringMatrix import PosScoringMatrix
from src.models.SVM_standard_kernels import get_kernel_choice
from src.utils import *
from src.features.build_features import *
from src.models.Estimator import metric_callable

# Datafile
data_file = "SIG_13.red.txt"

# Hyperparamaters
p = 14
q = 1


def choose_estimator(kernel=None, *args, **kwargs):
    est_label = []
    kernel_ = []
    estimator = None
    est_choice = 'Available estimators : \n' \
                 '\t - Position Scoring Matrix (to choose input: PSM) \n' \
                 '\t - Substitution Matrix (to choose input: SM) \n' \
                 '\t - sklearn-SVM (with multiple kernels) (to choose input: SVM) \n' \
                 'Enter estimator of choice: '
    while True:
        try:
            est_label = input(est_choice)
        except ValueError:
            print('Please enter a valid choice of estimator.')
        if est_label == 'PSM':
            estimator = PosScoringMatrix(*args, **kwargs)
            break
        elif est_label == 'SM':
            print('Substitution Matrix method not yet implemented.')
            continue
        elif est_label == 'SVM':
            if kernel is None:
                kernel_ = get_kernel_choice()
            else:
                kernel_ = kernel
            estimator = SVC(kernel=kernel_, *args, **kwargs)
            break
        elif est_label == 'quit':
            return None
        else:
            print('Please enter a valid choice of estimator.')
            continue

    return estimator


if __name__ == '__main__':
    # Hyperparameters
    C = 0.5
    params = [p, q, C]

    # Estimator parameters need to be filled in by hand
    estimator = choose_estimator(C=C, class_weight = 'balanced') # Change at will
    model = Model(estimator, params)

    # Getting features
    X, Y, _ = get_encoded_features(DATA_PATH + data_file, p, q)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Evaluating model
    print("Evaluating model...")
    score = model.evaluate(X_train, Y_train)
    print("Finished model evaliation.")
    estimator.fit(X_train, Y_train)
    pred = estimator.predict(X_test)
    for metric in METRIC_LIST:
        mc = metric_callable(metric)
        print("Classifier: " + model.estimator_name)
        print("{} score = {:.2f}".format(metric, mc(Y_test, pred)))


