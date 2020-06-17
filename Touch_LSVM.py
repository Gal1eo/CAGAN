# keystroke_ManhattanFiltered.py

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import numpy as np

np.set_printoptions(suppress=True)
import pandas
from EER import evaluateEER
from EER import evaluateFAR
import EER
from Touch_SVM import normalize_df


class LSVMDetector:
    # just the training() function changes, rest all remains same.

    def __init__(self, subjects, data, attacker_data):
        self.data = data
        self.attacker = attacker_data
        self.u_scores = []
        self.i_scores = []
        self.mean_vector = []
        self.subjects = subjects
        self.fp = []

    def training(self):
        self.clf = OneVsOneClassifier(SVC(kernel='rbf',gamma='auto'))
        labels = [0] * len(self.train) + [1] * len(self.train_imposter)
        self.clf.fit(pandas.concat([self.train,self.train_imposter]), labels)

    def testing(self):
        self.u_scores = self.clf.decision_function(self.test_genuine)
        self.i_scores = self.clf.decision_function(self.test_imposter)
        self.u_scores = list(self.u_scores)
        self.i_scores = list(self.i_scores)


    def evaluate(self):
        eers = []
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


                # self.test_imposter = normalize_np(self.attacker[idx])
                # self.test_imposter = normalize_df(imposter_data.groupby("user_id"). \
                #                                    head(10).loc[:,
                #                                    ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                #                                     'length of trajectory', 'mid-stroke pressure',
                #                                     'mid-stroke area covered',
                #                                     '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                #                                     '20\%-perc. dev. from end-to-end line',
                #                                     '50\%-perc. dev. from end-to-end line',
                #                                     '80\%-perc. dev. from end-to-end line']])
                self.train_imposter = normalize_df(imposter_data.groupby("user_id"). \
                                         tail(10).loc[:, ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                                                         'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
                                                         '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                                                         '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
                                                         '80\%-perc. dev. from end-to-end line']])


                self.test_imposter = self.attacker[idx]

                self.training()
                self.testing()
                # eers.append(evaluateEER(self.u_scores, \
                #                         self.i_scores))
                fpr.append(evaluateFAR(self.u_scores, self.i_scores))
                # print(evaluateFAR(self.u_scores, self.i_scores))

        else:
            genuine_user_data = self.data.loc[self.data.user_id == self.subjects, \
                                              ["stroke duration", 'start $x$', 'start $y$', 'stop $x$',
                                               'stop $y$',
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
            #                          tail(6).loc[:, "H.period":"H.Return"]
            # self.test_imposter = normalize_np(self.attacker)
            self.train_imposter = normalize_df(imposter_data.groupby("user_id"). \
                                         tail(10).loc[:, ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                                                         'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
                                                         '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                                                         '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
                                                         '80\%-perc. dev. from end-to-end line']])
            self.test_imposter = self.attacker

            self.training()
            self.testing()
            # eers.append(evaluateEER(self.u_scores, \
            #                        self.i_scores))
            fpr.append(evaluateFAR(self.u_scores, self.i_scores))

        return np.mean(fpr)


# user_path = "featMat.csv"
# data = pandas.read_csv(user_path)
# subjects = data["user_id"].unique()
# id = []
# for idx,subject in enumerate(subjects):
#     user_data = data.loc[data.user_id == subject, \
#              ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
#               'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered']]
#     #print(user_data.shape,idx)
#     if user_data.shape[0] >= 400:
#         id.append(subject,)
#
# #data = normalize(data)
#
# attacker_path = "attaker/data.csv"
#
#
# #subjects = "s002"
#
# #attacker_data = pandas.read_csv(attacker_path)
# #attacker_data =
#
#
# print("average EER for SVM detector:")
# print(LSVMDetector(id, data).evaluate())
