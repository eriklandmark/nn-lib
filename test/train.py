import datetime
import mnist
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_p(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost(A, y):
    return (y - A)**2

def cost_p(A, y):
    return 2 * (A - y)

def vectorize_label(z):
    output = np.zeros((1,10))
    output[0][z] = 1
    return output

def load_next_batch(data):
    images, labels = data.__next__()

    images = np.array(images)
    images = images / 255

    new_labels = []
    for label in labels:
        new_labels.append(vectorize_label(label))
    labels = np.vstack(new_labels)

    return images, labels

training_data = mnist.MNIST("data-set")

data = training_data.load_training_in_batches(10)

layer_sizes = [784, 32, 32, 10]

w1 = (np.random.random((layer_sizes[0], layer_sizes[1])) - 0.5) * 0.1
w2 = (np.random.random((layer_sizes[1], layer_sizes[2])) - 0.5) * 0.1
w3 = (np.random.random((layer_sizes[2], layer_sizes[3])) - 0.5) * 0.1

b1 = (np.random.random((1, layer_sizes[1])) - 0.5) * 0.1
b2 = (np.random.random((1, layer_sizes[2])) - 0.5) * 0.1
b3 = (np.random.random((1, layer_sizes[3])) - 0.5) * 0.1

learning_rate = 0.1

time = datetime.datetime.now()

#plt.imshow(np.reshape(images[1], (28, 28)), interpolation='nearest')
#plt.show()

for j in range(6000):
    images, labels = load_next_batch(data)
    for i in range(1000):
        z1 = images.dot(w1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(w2) + b2
        a2 = sigmoid(z2)
        z3 = a2.dot(w3) + b3
        a3 = sigmoid(z3)

        error = cost(a3, labels)
        gradient = cost_p(a3, labels)
        e3 = gradient * sigmoid_p(z3)
        e2 = (e3.dot(w3.T)) * sigmoid_p(z2)
        e1 = (e2.dot(w2.T)) * sigmoid_p(z1)

        b3 = b3 - (e3.mean(axis=0) * learning_rate)
        b2 = b2 - (e2.mean(axis=0) * learning_rate)
        b1 = b1 - (e1.mean(axis=0) * learning_rate)

        w3 = w3 - ((a2.T.dot(e3)) * learning_rate)
        w2 = w2 - ((a1.T.dot(e2)) * learning_rate)
        w1 = w1 - ((images.T.dot(e1)) * learning_rate)

    print(np.mean(error))

ann = np.array([b1, b2, b3, w1, w2, w3])
np.save("ann.npy", ann)

results = a3

for label in range(0, len(a3)):
    highest_index = 0
    highest_value = results[label][0]
    for index, result in enumerate(results[label]):
        if result > highest_value:
            highest_value = result
            highest_index = index

    print("Should find: {} | Did find: {}".format(list(labels[label]).index(1), highest_index))

print("Done in {}".format(datetime.datetime.now() - time))

#print(w1)
# print(w2)
# print(w3)