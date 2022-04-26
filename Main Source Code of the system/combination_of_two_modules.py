import numpy as np
import pandas as pd


data = pd.read_csv(r'C:\Users\Admin\Desktop\KLTN_NN\KLTN_NN_S\data.csv')


data = np.array(data)
num_of_cases, num_of_feature = data.shape

data_in = data[0:num_of_cases].T
Y_data = data_in[0]  # Labels
Y_data = Y_data.astype(int)
X_data = data_in[1:num_of_feature]


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
    return prediction


def get_prediction(prediction):
    return np.argmax(prediction, 0)


def get_paras_1st():
    weights_1st_1 = np.load('w1.npy')
    weights_1st_2 = np.load('w2.npy')
    biases_1st_1 = np.load('b1.npy')
    biases_1st_2 = np.load('b2.npy')

    return weights_1st_1, weights_1st_2, biases_1st_1, biases_1st_2


def get_paras_2nd(area):
    if area == 0:
        weights_2nd_1 = np.load('w1_0.npy')
        weights_2nd_2 = np.load('w2_0.npy')
        biases_2nd_1 = np.load('b1_0.npy')
        biases_2nd_2 = np.load('b2_0.npy')
    elif area == 1:
        weights_2nd_1 = np.load('w1_1.npy')
        weights_2nd_2 = np.load('w2_1.npy')
        biases_2nd_1 = np.load('b1_1.npy')
        biases_2nd_2 = np.load('b2_1.npy')
    elif area == 2:
        weights_2nd_1 = np.load('w1_2.npy')
        weights_2nd_2 = np.load('w2_2.npy')
        biases_2nd_1 = np.load('b1_2.npy')
        biases_2nd_2 = np.load('b2_2.npy')
    elif area == 3:
        weights_2nd_1 = np.load('w1_3.npy')
        weights_2nd_2 = np.load('w2_3.npy')
        biases_2nd_1 = np.load('b1_3.npy')
        biases_2nd_2 = np.load('b2_3.npy')
    elif area == 4:
        weights_2nd_1 = np.load('w1_4.npy')
        weights_2nd_2 = np.load('w2_4.npy')
        biases_2nd_1 = np.load('b1_4.npy')
        biases_2nd_2 = np.load('b2_4.npy')
    elif area == 5:
        weights_2nd_1 = np.load('w1_5.npy')
        weights_2nd_2 = np.load('w2_5.npy')
        biases_2nd_1 = np.load('b1_5.npy')
        biases_2nd_2 = np.load('b2_5.npy')
    elif area == 6:
        weights_2nd_1 = np.load('w1_6.npy')
        weights_2nd_2 = np.load('w2_6.npy')
        biases_2nd_1 = np.load('b1_6.npy')
        biases_2nd_2 = np.load('b2_6.npy')
    elif area == 7:
        weights_2nd_1 = np.load('w1_7.npy')
        weights_2nd_2 = np.load('w2_7.npy')
        biases_2nd_1 = np.load('b1_7.npy')
        biases_2nd_2 = np.load('b2_7.npy')

    return weights_2nd_1, weights_2nd_2, biases_2nd_1, biases_2nd_2


def predict(inputs, weight1, weight2, bias1, bias2):
    prediction = forward_propagation(weight1, weight2, bias1, bias2, inputs)
    predictions = get_prediction(prediction)
    return predictions


w_1st_1, w_1st_2, b_1st_1, b_1st_2 = get_paras_1st()
predict_area = predict(X_data, w_1st_1, w_1st_2, b_1st_1, b_1st_2)
pre_area = predict_area
# w_2nd_1, w_2nd_2, b_2nd_1, b_2nd_2 = get_paras_2nd(pre_area)
# predict_pos = predict(X_data, w_2nd_1, w_2nd_2, b_2nd_1, b_2nd_2)
size = len(predict_area)
for i in range(0, size):
    w_2nd_1, w_2nd_2, b_2nd_1, b_2nd_2 = get_paras_2nd(int(pre_area[i]))
    # print(predict_area[i])
    predict_pos = predict(X_data, w_2nd_1, w_2nd_2, b_2nd_1, b_2nd_2)
    print(predict_pos[i])
    i = i + 1

'''
print("Area: ")
for x in range (0, len(predict_area)):
    print(predict_area[x], "\r")
    x = x + 1
print("Position: ")
for y in range (0, len(predict_pos)):
    print(predict_pos[y], "\r")
    y = y + 1
'''
