import pandas

user_path = "featMat.csv"
data = pandas.read_csv(user_path)
subjects = data["user_id"].unique()
id = []
for idx,subject in enumerate(subjects):
    user_data = data.loc[data.user_id == subject, \
             ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
             'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
             '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
             '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
             '80\%-perc. dev. from end-to-end line']]
    print(user_data.shape,idx)
    if user_data.shape[0] >= 400:
        id.append(subject,)

# final = data.filter(axis=1 ,items=["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
#              'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
#              '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
#              '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
#              '80\%-perc. dev. from end-to-end line'])

final = data.loc[data.user_id.isin(id), \
             ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
             'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
             '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
             '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
             '80\%-perc. dev. from end-to-end line']]


print(final.shape)
print(id)