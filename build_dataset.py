import random
import pickle


with open('.../remap.pkl', 'rb') as f:
    user = pickle.load(f)
    movie = pickle.load(f)
    rating = pickle.load(f)

# construct timestamp and movie_label data_set
data_set = []

for UserID, hist in rating.groupby('UserID'):
    nearest_watch_time = hist['Timestamp'].max()
    hist_row = hist.loc[hist[hist['Timestamp'] == nearest_watch_time].index].reset_index(drop=True)
    hist_row_sub = hist_row.loc[0]
    data_set.append((list(hist_row['MovieID']), int(hist_row_sub['Timestamp'])))
# construct user_vector
train_data = []

user.apply(lambda row: train_data.append(
    (int(row['Gender']), int(row['Occupation']), float(row['Age']), int(row['Zip-code']), row['Watch_History'])),
           axis=1)

training_data = list(i+j for i, j in zip(train_data, data_set))

random.shuffle(training_data)

with open('.../dataset.pkl') as f:
    pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
