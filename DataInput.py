import numpy as np


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.batch_num_per_epoch = len(data) // self.batch_size
        if self.batch_num_per_epoch * self.batch_size < len(data):
            self.batch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.batch_num_per_epoch:
            raise StopIteration

        batch_data = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]

        self.i += 1

        gender, occupation, age, geographic, watch_history, timestamp, y = [], [], [], [], [], [], []
        for item in batch_data:
            gender.append(item[0])
            occupation.append(item[1])
            age.append(item[2])
            geographic.append(item[3])
            watch_history.append(len(item[4]))
            timestamp.append(item[6])
            y.append(item[5])
        max_watch_history = max(watch_history)

        watch_history_matrix = np.zeros([len(batch_data), max_watch_history], np.int64)

        row_count = 0
        for item in batch_data:
            for l in range(len(item[4])):
                watch_history_matrix[row_count][l] = item[4][l]
            row_count += 1
        return self.i, (gender, occupation, age, geographic, watch_history_matrix, timestamp, y)
