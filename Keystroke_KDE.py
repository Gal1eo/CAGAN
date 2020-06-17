import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from Keystroke_SVM import SVMDetector
from Keystroke_LSVM import LSVMDetector
from Keystroke_GMM import GMMDetector
import time

user_path = "keystroke.csv"
data = pd.read_csv(user_path)
subjects = data["subject"].unique()
far1 = []
far2= []
far3 = []
tic = time.time()

for subject in subjects:
    params = {'bandwidth': np.logspace(-1, 1, 40)}
    grid = GridSearchCV(KernelDensity(), params)
    subject_data = data.loc[data.subject == subject, "H.period":"H.Return"]
    train_data = subject_data[:200]
    grid.fit(train_data)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    new_data = kde.sample(250)
    #print(SVMDetector(subject, data, new_data).evaluate())
#     far1.append(SVMDetector(subject, data, new_data).evaluate())
#     far2.append(LSVMDetector(subject, data, new_data).evaluate())
#     far3.append(GMMDetector(subject, data, new_data).evaluate())
#
# print(np.mean(far1))
# print(np.mean(far2))
# print(np.mean(far3))
toc = time.time()

print(toc - tic)