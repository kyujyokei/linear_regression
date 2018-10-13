import numpy as np
import math
import csv


def load_file(filename, has_label=True):
    data = np.genfromtxt(filename, dtype=np.str, delimiter=",")

    data = np.delete(data, 0, axis=0)
    data = np.delete(data, 1, axis=1)

    data = np.insert(data, 2, values='year', axis=1)
    data = np.insert(data, 2, values='day', axis=1)
    data = np.insert(data, 2, values='month', axis=1)

    # read data

    for i in range(len(data)):
        # print(data[i])
        date = data[i][1].split('/')
        data[i][2], data[i][3], data[i][4] = date[0], date[1], date[2]

        if data[i][17] == '0':
            data[i][17] = data[i][16]

        if data[i][18][:3] == '981':
            data[i][18] = 1
        else:
            data[i][18] = 0

    data = np.delete(data, 1, axis=1)
    data = data.astype(float)
    label = data[:, -1:]
    data = data[:, :-1]

    print("Waterfront:  ", calc_percentage(data[:, 9]))
    print("Condition:   ", calc_percentage(data[:, 11]))
    print("Grade:       ", calc_percentage(data[:, 12]))
    print("ZIP Code:    ", calc_percentage(data[:, 17]))

    with open(filename+'stats.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # writer.writerow(['John Smith', 'Accounting', 'November'])
        # writer.writerow(['Erica Meyers', 'IT', 'March'])
        writer.writerow(['','Mean', 'Standard Deviation', 'Range', 'Min', 'Max'])

        for k in range(0,22):
            if k != 9 or k != 11 or k != 12 or k!= 17:
                # print("CALC = ",k)
                # print_stats(data[:, k])
                writer.writerow([k,
                                 np.mean(data[:, k], axis=0),
                                 np.std(data[:, k], axis=0),
                                 np.ptp(data[:, k], axis=0),
                                 np.min(data[:, k], axis=0),
                                 np.max(data[:, k], axis=0)])


    if has_label:
        return data, label
    else:
        return data

def print_stats(data):
    print("Mean: ", np.mean(data, axis=0))
    print("Standard Deviation: ", np.std(data, axis=0))
    print("Range: ", np.ptp(data, axis=0))
    print("Min", np.min(data, axis=0))
    print("Max", np.max(data, axis=0))


def calc_percentage(data):
    dic = {}
    for i in data:
        dic[i] = dic[i] + 1 if i in dic else 1
    for  j in dic:
        dic[j] = format((j == data).sum() / len(data), '.3f')
    return dic


# print("Mean: ", np.mean(data, axis=0))
# print("Standard Deviation: ", np.std(data, axis=0))
# print("Range: ", np.ptp(data, axis=0))
# print(np.min(data, axis=0))
# print(np.max(data, axis=0))
# print("Waterfront = 1 : ", (0 < data[:, 9]).sum() / len(data[:, 9]))
# print("", (0 == data[:, 9]).sum() / len(data[:, 9]))


# print("", np.percentile(data, 1))
# print(data[0])




""" normalization"""

# for i in range(len(data[0])-1):
#     data[:, i] = data[:, i] / np.linalg.norm(data[:, i])
# # print(data[0])

def normalize(v):
    """
    normalize a vector
    :param v:
    :return:
    """
    if np.ptp(v) == 0:
        return v / v[0]
    return (v - v.min()) / (np.ptp(v))
    # norm = np.linalg.norm(v)
    # if norm == 0:
    #    return v
    # return v / norm


def normalize_matrix(data):
    """
    normalize a matrix according to each column
    :param data: matrix
    :return:
    """
    if data is None or len(data) == 0:
        return data
    for i in range(len(data[0])):
        data[:, i] = normalize(data[:, i])
    return data


data, label = load_file("PA1_train.csv")
data = normalize_matrix(data)

data_t = np.transpose(data)
# print(data_t[1])


lr = 0.000001
lam = 0.1

w = np.zeros((len(data[0]), 1))

for iter in range(10000):

    predict_y = np.matmul(data, w)

    grad = np.matmul(data_t, label - predict_y) + lam * w

    w += lr * grad

    if iter == 99999:
        # print("W: ", w)
        norm = 0
        for val in grad:
            norm += val ** 2
        norm = math.sqrt(norm)

        # print()
        print("Norm: ", norm)

train_predict = np.matmul(data, w)

# print("W: ", w)

train_sse = (((label - train_predict)**2).sum())

print("Train SSE: ", train_sse)

print("=======================================================")

valid_data, valid_label = load_file("PA1_dev.csv")
valid_data = normalize_matrix(valid_data)

dev_predict = np.matmul(valid_data, w)

dev_sse = (((valid_label - dev_predict)**2).sum())

print("=======================================================")

test_data = load_file("PA1_test.csv")


print("Dev SSE: ", dev_sse)


