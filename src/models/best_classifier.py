from sklearn.svm import SVC
from src.utils import *
from src.features.build_features import *


# Datafile
train_data_file = "SIG_13.red.txt"
test_data_file = "test_data.txt"


if __name__ == '__main__':
    # Hyperparameters
    C = 0.5
    p = 14
    q = 1

    # Getting training and testing data
    X_train, Y_train, _ = get_encoded_features(DATA_PATH + train_data_file, p, q)
    X_test, predicted_pos = get_encoded_features(DATA_PATH + test_data_file, p, q, labels=False)

    # Best estimator
    estimator = SVC(C=C, kernel='rbf', class_weight='balanced')
    estimator.fit(X_train, Y_train)
    binary_predictor = estimator.predict(X_test)

    # Output prediction
    print("Classifier: SVM with RBF kernel, C parameter = 0.5 and default gamma value.\n")
    for i in range(len(binary_predictor)):
        if binary_predictor[i] == 1:
            print("Predicted cleavage site position on sequence {} : {}".format(predicted_pos[i][0],
                                                                                predicted_pos[i][1]))




