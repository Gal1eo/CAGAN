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
    subject_data = data.loc[data.subject == subject, "H.period":"H.Return"]
    train_data = subject_data[:40]
    mean_of_train = np.mean(np.single(train_data.values), axis=0)
    new_data = np.random.normal(mean_of_train, 0.1, (250, 31))



    # far1.append(SVMDetector(subject, data, new_data).evaluate())
    # far2.append(LSVMDetector(subject, data, new_data).evaluate())
    # far3.append(GMMDetector(subject, data, new_data).evaluate())

# print(np.mean(far1))
# print(np.mean(far2))
# print(np.mean(far3))
toc = time.time()
print(toc - tic)