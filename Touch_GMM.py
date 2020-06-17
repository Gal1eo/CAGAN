# keystroke_GMM.py

from sklearn.mixture import GaussianMixture
import pandas
from EER_GMM import evaluateEERGMM
from EER import evaluateFAR
import numpy as np
import warnings


warnings.filterwarnings("ignore")
from Touch_SVM import normalize_df

class GMMDetector:
    # the training(), testing() and evaluateEER() function change, rest all is same.

    def __init__(self, subjects, data, attacker_data):
        self.data = data
        self.attacker = attacker_data
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects

    def training(self):
        self.gmm = GaussianMixture(n_components=2, covariance_type='diag',
                                   verbose=False)
        self.gmm.fit(self.train)

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values
            j = j.reshape(1, -1)
            cur_score = self.gmm.score(j)
            self.user_scores.append(cur_score)

        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter[i]
            # j = self.test_imposter.iloc[i].values
            j = j.reshape(1, -1)
            cur_score = self.gmm.score(j)
            self.imposter_scores.append(cur_score)

    def evaluate(self):
        fpr = []

        if isinstance(self.subjects, list):
            for idx, subject in enumerate(self.subjects):
                genuine_user_data = self.data.loc[self.data.user_id == subject, \
                                                  ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                                                   'length of trajectory', 'mid-stroke pressure',
                                                   'mid-stroke area covered',
                                                   '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                                                   '20\%-perc. dev. from end-to-end line',
                                                   '50\%-perc. dev. from end-to-end line',
                                                   '80\%-perc. dev. from end-to-end line']]
                imposter_data = self.data.loc[self.data.user_id != subject, :]
                # generated_data = attacker_data
                genuine_user_data = normalize_df(genuine_user_data[:400])

                self.train = genuine_user_data[:200]
                self.test_genuine = genuine_user_data[200:400]
                # self.test_imposter = normalize(imposter_data.groupby("user_id"). \
                #                          tail(10).loc[:, ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                #                                          'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
                #                                          '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                #                                          '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
                #                                          '80\%-perc. dev. from end-to-end line']])
                # print(idx)
                # self.test_imposter = normalize_np(self.attacker[idx])
                self.test_imposter = self.attacker[idx]
                self.training()
                self.testing()
                # eers.append(evaluateEER(self.u_scores, \
                #                         self.i_scores))
                fpr.append(evaluateEERGMM(self.user_scores, self.imposter_scores))
                # print(evaluateFAR(self.u_scores, self.i_scores))


        else:
            genuine_user_data = self.data.loc[self.data.user_id == self.subjects, \
                                              ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                                               'length of trajectory', 'mid-stroke pressure',
                                               'mid-stroke area covered',
                                               '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                                               '20\%-perc. dev. from end-to-end line',
                                               '50\%-perc. dev. from end-to-end line',
                                               '80\%-perc. dev. from end-to-end line']]
            imposter_data = self.data.loc[self.data.user_id != self.subjects, :]
            # generated_data = attacker_data
            genuine_user_data = normalize_df(genuine_user_data[:400])

            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:400]
            # self.test_imposter = imposter_data.groupby("subject"). \
            #                          head(6).loc[:, "H.period":"H.Return"]
            self.test_imposter = self.attacker

            self.training()
            self.testing()
            fpr.append(evaluateEERGMM(self.user_scores, self.imposter_scores))

        return np.mean(fpr)


'''
path = "keystroke.csv"
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects,data).evaluate())
'''
