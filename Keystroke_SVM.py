#keystroke_ManhattanFiltered.py

from sklearn.svm import OneClassSVM
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER
from EER import evaluateFAR
import EER


class SVMDetector:
#just the training() function changes, rest all remains same.

    def __init__(self, subjects, data, attacker_data):
        self.data = data
        self.attacker = attacker_data
        self.u_scores = []
        self.i_scores = []
        self.mean_vector = []
        self.subjects = subjects
        self.fp = []

    def training(self):
        self.clf = OneClassSVM(kernel='rbf',gamma=26)
        self.clf.fit(self.train)
 
    def testing(self):
        self.u_scores = -self.clf.decision_function(self.test_genuine)
        self.i_scores = -self.clf.decision_function(self.test_imposter)
        self.u_scores = list(self.u_scores)
        self.i_scores = list(self.i_scores)
        #self.predict = self.clf.predict(self.test_imposter)
        #self.fp = np.count_nonzero(self.predict==1)/250
        #self.fp = list(self.predict)
 
    def evaluate(self):
        eers = []
        fpr = []

        if len(self.subjects) > 4:
            for idx,subject in enumerate(self.subjects):
                genuine_user_data = self.data.loc[self.data.subject == subject, \
                                             "H.period":"H.Return"]
                imposter_data = self.data.loc[self.data.subject != subject, :]
                #generated_data = attacker_data

                self.train = genuine_user_data[:200]
                self.test_genuine = genuine_user_data[200:]
                # self.test_imposter = imposter_data.groupby("subject"). \
                #                      tail(6).loc[:, "H.period":"H.Return"]
                self.test_imposter = self.attacker[idx]

                self.training()
                self.testing()
                # eers.append(evaluateEER(self.u_scores, \
                #                          self.i_scores))
                fpr.append(evaluateFAR(self.u_scores,self.i_scores))

        else:
            genuine_user_data = self.data.loc[self.data.subject == self.subjects, \
                                "H.period":"H.Return"]
            imposter_data = self.data.loc[self.data.subject != self.subjects, :]
            # generated_data = attacker_data

            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            # self.test_imposter = imposter_data.groupby("subject"). \
            #                          tail(6).loc[:, "H.period":"H.Return"]
            self.test_imposter = self.attacker

            self.training()
            self.testing()
            # eers.append(evaluateEER(self.u_scores, \
            #                         self.i_scores))
            fpr.append(evaluateFAR(self.u_scores, self.i_scores))


        return np.mean(fpr)

'''
user_path = "keystroke.csv"

attacker_path = "attaker/data.csv"
data = pandas.read_csv(user_path)
subjects = data["subject"].unique()
#subjects = "s002"

attacker_data = pandas.read_csv(attacker_path)
print("average EER for SVM detector:")
print(SVMDetector(subjects[1], data,attacker_data).evaluate())
'''
