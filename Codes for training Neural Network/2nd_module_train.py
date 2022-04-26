import numpy as np
import pandas as pd
import csv


data = pd.read_csv(r'C:\Users\Admin\Desktop\KLTN_NN\KLTN_SNN_0\data.csv')

test = pd.read_csv(r'C:\Users\Admin\Desktop\KLTN_NN\KLTN_SNN_0\test.csv')

test = np.array(test)
n, m = test.shape


data_test = test[0:n].T
Y_test = data_test[0]
Y_test = Y_test.astype(int)
X_test = data_test[1:m]


data = np.array(data)
num_of_cases, num_of_feature = data.shape

np.random.shuffle(data)

data_train = data[0:num_of_cases].T
Y_train = data_train[0]  # Labels
Y_train = Y_train.astype(int)
X_train = data_train[1:num_of_feature]


def init_parameters():
    # 16 inputs, 16 neurons in layer 1, 40 neurons in layer 2
    inw1 = np.random.rand(16, 16)  # weights of layer 1 is (16x16) shape
    inb1 = np.random.rand(16, 1)   # biases of layer 1 is (16x1) shape
    inw2 = np.random.rand(40, 16)   # weights of layer 2 is (40x16) shape
    inb2 = np.random.rand(40, 1)    # biases of layer 2 is (40x1) shape
    return inw1, inw2, inb1, inb2


def relu(inputs):
    return np.maximum(inputs, 0)


def softmax(inputs):
    exp_value = np.exp(inputs - np.max(inputs))
    probabilities = exp_value / np.sum(exp_value, axis=0, keepdims=True)
    return probabilities


def forward_propagation(weight1, weight2, bias1, bias2, inputs):
    z1 = weight1.dot(inputs) + bias1
    output1 = relu(z1)
    z2 = weight2.dot(output1) + bias2
    prediction = softmax(z2)
    return prediction, output1, z1, z2


def relu_back(inputs):
    return inputs > 0


def one_hot(inputs):  # one_hot encoding the label (true result)
    one_hot_inputs = np.zeros((inputs.size, int(inputs.max()) + 1), dtype=int)
    one_hot_inputs[np.arange(inputs.size), inputs] = 1
    one_hot_inputs = one_hot_inputs.T
    return one_hot_inputs


def backward_propagation(prediction, output1, weight2, inputs, result):
    one_hot_result = one_hot(result)  # one_hot encoding the correct result
    dz2 = prediction - one_hot_result  # error calculation
    dw2 = (1 / num_of_cases) * dz2.dot(output1.T)
    db2 = (1 / num_of_cases) * np.sum(dz2)

    dz1 = weight2.T.dot(dz2) * relu_back(output1)
    dw1 = (1 / num_of_cases) * dz1.dot(inputs.T)
    db1 = (1 / num_of_cases) * np.sum(dz1)
    return dw1, dw2, db1, db2


def update_parameters(weight_1, weight_2, bias_1, bias_2, dw1, dw2, db1, db2, learning_rate):
    weight_1 = weight_1 - learning_rate * dw1
    bias_1 = bias_1 - learning_rate * db1
    weight_2 = weight_2 - learning_rate * dw2
    bias_2 = bias_2 - learning_rate * db2
    return weight_1, weight_2, bias_1, bias_2


def get_prediction(prediction):
    return np.argmax(prediction, 0)


def get_accuracy(prediction, result):

    print('Prediction: ', prediction)
    print('True result: ', result)
    return np.sum(prediction == result) / result.size


def training(inputs, result, iterations, learning_rate):
    weights1 = np.load('w1_6.npy')
    weights2 = np.load('w2_6.npy')
    biases1 = np.load('b1_6.npy')
    biases2 = np.load('b2_6.npy')
    a = open('a.csv', 'a')
    # weights1, weights2, biases1, biases2 = init_parameters()
    for i in range(iterations):
        prediction, output1, z1, z2 = forward_propagation(weights1, weights2, biases1, biases2, inputs)
        dw1, dw2, db1, db2 = backward_propagation(prediction, output1, weights2, inputs, result)
        weights1, weights2, biases1, biases2 = update_parameters(weights1, weights2, biases1, biases2, dw1, dw2, db1, db2, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_prediction(prediction )
            acc = get_accuracy(predictions, result)
            a.write(str(acc))
            a.write('\n')
            print(predictions)
            print(acc)
    a.close()
    return weights1, weights2, biases1, biases2


def test(inputs, result, weight1, weight2, bias1, bias2):
    prediction, output1, z1, z2 = forward_propagation(weight1, weight2, bias1, bias2, inputs)
    predictions = get_prediction(prediction)
    accuracies = get_accuracy(predictions, result)
    return predictions, accuracies


w1, w2, b1, b2 = training(X_train, Y_train, 10000, 0.3)
print("Test")
pre, acc = test(X_test, Y_test, w1, w2, b1, b2)
np.save('w1_6.npy', w1)
np.save('w2_6.npy', w2)
np.save('b1_6.npy', b1)
np.save('b2_6.npy', b2)
print("Accuracy = ", acc)

'''

writer_a = csv.writer(a)
writer_a.writerow(acc)

'''