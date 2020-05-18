from src.models.Estimator import Estimator, metric_callable
from src.features.build_features import *
from src.models.Model import Model
from src.utils import *
from src.visualization.heatmap import *
from sklearn.model_selection import train_test_split

# Hyperparameters
p_max = 16
q_max = 16
p = 15
q = 1
alpha = 0.5
C = -58
cv = 5
test_size = 0.2


def add_smoothing(num, denom, alpha=1.):
    return (num + alpha) / (denom + alpha)


def psm_accuracy(cleav_test_batch, cleav_predicted_batch):
    accuracy = 0
    if len(cleav_predicted_batch) > 0:
        for i in range(len(cleav_test_batch)):
            if i in cleav_predicted_batch:
                for prev in cleav_predicted_batch[i]:
                    if prev == cleav_test_batch[i]:
                        accuracy += 1
        accuracy /= len(cleav_predicted_batch)
    return 100 * accuracy


class PosScoringMatrix(Estimator):
    """
    Class that implements an estimator of cleavage site location in protein sequences by the weight-matrix method,
    detailed in [1]. Inherits the Estimator class.

    Suggested Changes
    -----------------
    - Normalize the threshold parameter C to the interval [0,1]
    - Generalize ignore_first to ignore a list of positions passed as a parameter

    References
    ----------
    [1] von Heijne, Gunnar. “A New Method for Predicting Signal Sequence Cleavage Sites.”
    Nucleic Acids Research, vol. 14, no. 11, 1986, pp. 4683–90. DOI.org (Crossref), doi:10.1093/nar/14.11.4683.
    """

    def __init__(self, p, q, C, alpha=0.5, ignore_first=True, write_log=True):
        """
        Parameters
        ----------
        p: int
            One of the hyperparameters that controls the subsequence length. Corresponds to the number of
            aminoacids to the left of the predicted cleavage site.
        q: int
            One of the hyperparameters that controls the subsequence length. The value q - 1 corresponds to the number
            of aminoacids to the right of the predicted cleavage site.
        C: float
            Threshold for the binary classifier. It is currently not normalized, so its optimal value should be tested
            between the interval [-70, -50] depending on the training data.
        alpha: float, defaults to 0.5
            Parameter for the additive smoothing method used for computing the score matrix with pseudocounts.
        ignore_first: bool, defaults to True
            Whether to ignore the first amino-acid in every sequence, which generally corresponds to the same starting
            letter. Setting its value to False may introduce bias in the model.
        write_log: bool, defaults to True
            Whether to write the scoring results in PredictionLogs text file.
        """

        super().__init__(p, q, write_log)
        self.C = C
        self.alpha = alpha
        self.ignore_first = ignore_first
        self.alphabet = None
        self.d = None
        self.score_matrix = None

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def word_score(self, sequence, lbound, ubound):
        # Computes the score of a subsequence. Auxiliary function for the predict() method.

        score_sum = 0
        cont = 0
        for i in range(lbound, ubound):
            score_sum += self.score_matrix[self.d[sequence[i]], cont]
            cont += 1

        return score_sum

    def get_test_labels(self, seq_test_batch, cleav_test_batch, crit='exact'):
        # Transforms the list of cleavage sites into a {+1, -1} list for use in the score function.
        exact_comp = lambda target, start: (start + self.p) == target
        neighbourhood_comp = lambda target, start: start <= target <= (start + self.p + self.q)

        if crit == 'neighbour':
            comp = neighbourhood_comp
        else:
            comp = exact_comp

        test_labels = []

        seq_cont = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                if comp(cleav_test_batch[seq_cont], i):
                    test_labels.append(1)
                else:
                    test_labels.append(-1)
                i += 1
            seq_cont += 1

        return test_labels

    def fit(self, seq_train_batch, cleav_train_batch):
        # Computes the weight matrix relative to the training data seq_train_batch and their corresponding
        # cleavage sites cleav_train_batch.

        super().fit(seq_train_batch, cleav_train_batch)

        if self.alphabet is None:
            self.alphabet = return_alphabet(seq_train_batch)

        if self.d is None:
            self.d = dict_from_alphabet(self.alphabet)

        dim = len(self.d)
        N = len(seq_train_batch)

        # Filling up matrix with observed frequencies
        score = np.zeros((dim, self.p + self.q))
        for letter in self.alphabet:
            seq_cont = 0
            for seq in seq_train_batch:
                if self.ignore_first:
                    seq = seq[1:]

                seq_len = len(seq)
                cp = cleav_train_batch[seq_cont]
                max_iter = min(self.p + self.q, seq_len + self.p - cp)
                for i in range(max_iter):
                    if seq[cp - self.p + i] == letter:
                        score[self.d[letter], i] += 1
                seq_cont += 1

        # Additive smoothing
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = add_smoothing(score[a, i], N, self.alpha)

        # Computing total frequency of each letter
        total_freq = np.zeros(dim)
        for letter in self.alphabet:
            for seq in seq_train_batch:
                for seq_letter in seq:
                    if seq_letter == letter:
                        total_freq[self.d[letter]] += 1

        total_freq = np.log(total_freq / N)

        # Taking the logarithm
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = np.log(score[a, i]) - total_freq[a]

        self.score_matrix = score

    def predict(self, seq_test_batch):
        # Performs prediction of a test batch and returns +1 or -1 labels for each subsequence.
        # Used in the score method.

        super().predict(seq_test_batch)

        cleav_predictor = []

        seq_cont = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                ws = self.word_score(test_seq, i, i + self.p + self.q)
                if ws > self.C:
                    cleav_predictor.append(1)
                else:
                    cleav_predictor.append(-1)
                i += 1
            seq_cont += 1

        return cleav_predictor

    def score(self, seq_list, cleavpos, cv=5, scoring=METRIC_LIST):
        # Uses super-class Estimator's built-in score method
        return super().score(seq_list, cleavpos, cv=cv, scoring=scoring)


if __name__ == "__main__":
    # Data processing
    data_file = "SIG_13.red.txt"
    seq_list, cleavpos = get_features(DATA_PATH + data_file)
    X_train, X_test, Y_train, Y_test = train_test_split(seq_list, cleavpos, test_size=test_size, random_state=0)

    # Defining p x q grid
    p_list = np.arange(p_max - 1)
    q_list = np.arange(q_max - 1)
    matrix_dict = dict(zip(METRIC_LIST, [np.zeros((p_max - 1, q_max - 1)) for i in range(len(METRIC_LIST))]))

    # Model training and assessment
    for pp in p_list:
        for qq in q_list:
            estimator = PosScoringMatrix(pp + 1, qq + 1, C, alpha)
            model = Model(estimator, [pp + 1, qq + 1, C], cv)
            score = model.evaluate(X_train, Y_train)
            for metric in METRIC_LIST:
                matrix_dict[metric][pp][qq] = score[metric]

    # Plotting heatmaps
    p_string = ["p=" + str(p + 1) for p in p_list]
    q_string = ["q=" + str(q + 1) for q in q_list]
    for metric in METRIC_LIST:
        fig, ax = plt.subplots()

        im, cbar = heatmap(matrix_dict[metric], p_string, q_string, ax=ax,
                           cmap="YlGn", cbarlabel="Accuracy heatmap, metric={}".format(metric))
        texts = annotate_heatmap(im, valfmt="{x:.2f}")

        fig.tight_layout()
        plt.show()
