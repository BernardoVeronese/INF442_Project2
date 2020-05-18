from abc import ABC

from src.models.Estimator import metric_callable
from src.features.build_features import *
from src.models.Model import Model
from src.utils import *
from src.visualization.heatmap import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data_file = "SIG_13.red.txt"


def add_smoothing(num, denom, alpha=1.):
    return (num + alpha) / (denom + alpha)


class ProbKernel:
    """
    Class that implements the probability kernel described in [1], using functions from the PosScoringMatrix class.

    Its most important method is return_kernel_matrix, which returns a matrix that can be used as a custom kernel for
    the SVC class.

    Suggested Changes
    -----------------
    - Normalize the threshold parameter C to the interval [0,1]
    - Generalize ignore_first to ignore a list of positions passed as a parameter

    References
    ----------
    [1] Vert, J. P. “SUPPORT VECTOR MACHINE PREDICTION OF SIGNAL PEPTIDE CLEAVAGE SITE USING A NEW CLASS OF KERNELS
    FOR STRINGS.” Biocomputing 2002, WORLD SCIENTIFIC, 2001, pp. 649–60. DOI.org (Crossref),
    doi:10.1142/9789812799623_0060.
    """

    def __init__(self, p, q, alpha=0.5, ignore_first=True):
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
        """
        self.p = p
        self.q = q
        self.alpha = alpha
        self.ignore_first = ignore_first
        self.alphabet = None
        self.d = None
        self.score_matrix = None
        self.kernel_matrix = None

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

        """# Computing total frequency of each letter
        total_freq = np.zeros(dim)
        for letter in self.alphabet:
            for seq in seq_train_batch:
                for seq_letter in seq:
                    if seq_letter == letter:
                        total_freq[self.d[letter]] += 1

        total_freq = total_freq / N

        # Taking the logarithm
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = score[a, i] / total_freq[a]
                """

        self.score_matrix = score

    def pairwise_kernel(self, seq_a, seq_b, ia, ib):
        # Calculates kernel score between two words given by the pair (sequence, index at sequence).

        if self.score_matrix is None:
            raise ValueError("Fit method must be executed first")

        kernel = 1.

        for iter in range(self.p + self.q):
            aux = self.score_matrix[self.d[seq_a[iter + ia]]][iter]
            if seq_a[iter + ia] == seq_b[iter + ib]:
                kernel *= aux + aux * aux / 2
            else:
                kernel *= aux * self.score_matrix[self.d[seq_b[iter + ib]]][iter] / 2

        return kernel

    def return_kernel_matrix(self, seq_train_batch, seq_test_batch=None):
        # Computes kernel matrix for use in the sklearn.svm.SVC() class.
        if seq_test_batch is None:
            seq_test_batch = seq_train_batch

        wl = self.p + self.q
        n_seq_train = len(seq_train_batch)
        n_seq_test = len(seq_test_batch)
        if not self.ignore_first:
            km_len_train = sum([len(seq) for seq in seq_train_batch]) - n_seq_train * wl
            km_len_test = sum([len(seq) for seq in seq_test_batch]) - n_seq_test * wl
        else:
            km_len_train = sum([len(seq) for seq in seq_train_batch]) - n_seq_train * (wl + 1)
            km_len_test = sum([len(seq) for seq in seq_test_batch]) - n_seq_test * (wl + 1)

        self.kernel_matrix = np.ones((km_len_test, km_len_train)) * (-1)

        cont_test = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            ntest = len(test_seq)
            itest = 0
            while itest < ntest - wl:
                cont_train = 0
                for train_seq in seq_train_batch:
                    if self.ignore_first:
                        train_seq = train_seq[1:]

                    ntrain = len(train_seq)
                    itrain = 0
                    while itrain < ntrain - wl:
                        if self.kernel_matrix[cont_test + itest, cont_train + itrain] < -0.9:
                            self.kernel_matrix[cont_test + itest, cont_train + itrain] = self.pairwise_kernel(test_seq,
                                                                                                              train_seq,
                                                                                                              itest,
                                                                                                              itrain)
                        itrain += 1
                    cont_train += ntrain - wl
                itest += 1
            cont_test += ntest - wl

        return self.kernel_matrix


if __name__ == "__main__":
    # Hyperparameters
    p = 3
    q = 1
    C = 1

    # Training and testing batches, small values for function verification
    ntrain = 50
    ntest = 20

    seq_list, cleav_pos = get_features(DATA_PATH + data_file)
    alphabet = return_alphabet(seq_list)

    X, Y, predicted_sites = get_encoded_features(DATA_PATH + data_file, p, q)
    X_train, Y_train, _ = seq_list_encoding(seq_list[:ntrain], cleav_pos[:ntrain], p, q, alphabet)
    X_test, Y_test, _ = seq_list_encoding(seq_list[+ntrain:+ntrain+ntest], cleav_pos[+ntrain:+ntrain+ntest], p, q, alphabet)

    prob_kernel = ProbKernel(p, q)
    prob_kernel.fit(seq_list[:ntrain], cleav_pos[:ntrain])
    km = prob_kernel.return_kernel_matrix(seq_list[:ntrain], seq_list[:ntrain])
    estimator = SVC(C=10, kernel='precomputed', class_weight='balanced')
    estimator.fit(km, Y_train)
    km = prob_kernel.return_kernel_matrix(seq_list[:ntrain], seq_list[ntrain:ntrain + ntest])
    pred_y = estimator.predict(km)

    for metric in METRIC_LIST:
        mc = metric_callable(metric)
        print("({}, {})".format(metric, mc(Y_test, pred_y)))

