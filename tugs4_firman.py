import numpy as np
import collections
from sklearn.naive_bayes import GaussianNB

# membaca data set dan data test
data_set = np.loadtxt("TrainsetTugas4ML.csv", skiprows=1, delimiter=",")
data_test = np.genfromtxt("TestsetTugas4ML.csv", delimiter=",")[1:-1]


def Bootstrap(dataset):
    bootstrap = np.zeros(dataset.shape)
    for i in range(dataset.shape[0]):
        idx = np.random.randint(dataset.shape[0])
        bootstrap[i] = dataset[idx]
    return bootstrap


def Split(data):
    data_input = data.T[:2].T
    data_class = data.T[-1].T
    return data_input, data_class


def Model(bootstrap):
    input, target = Split(bootstrap)
    model = GaussianNB().fit(input, target)
    return model


jumlah_bootstrap = 27
bootstraps = np.zeros((jumlah_bootstrap, data_set.shape[0], data_set.shape[1]))

# Create bootstrap and Train models
models = []
for i in range(bootstraps.shape[0]):
    bootstraps[i] = Bootstrap(data_set)
    models.append(Model(bootstraps[i]))


# Predict all data test
input, target = Split(data_test)
outputs = []
for model in models:
    output = model.predict(input)
    outputs.append(output)
outputs = np.array(outputs)


# Let's Vote
dua = 0
satu = 0
tebakan = []
for i, output in enumerate(outputs.T):
    count = collections.Counter(output)
    most = count.most_common(1)[0][0]

    data_test[i][2] = most
    tebakan.append(most)

    if data_test[i][2] == 2:
        dua += 1
    elif data_test[i][2] == 1:
        satu += 1

print("Jumalah Berlabel 2 = ", dua)
print("Jumalah Berlabel 1 = ", satu)
print("Hasil Berada di file TebakanTugas4ML.csv")

# Save output
np.savetxt("TebakanTugas4ML.csv", tebakan, delimiter=",")

