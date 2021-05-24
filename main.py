import random

def read_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        data_set = []
        for line in lines:
            line = line.strip()
            parts = line.split(',')
            x0, x1, y = float(parts[0]), float(parts[1]), float(parts[2])
            data_set.append((x0, x1, y))
        
        return data_set

def f(x):
    return x

def df(x):
    return 1

def feed_forward_info(x0, x1, perceptron, activation_function):
    w0 = perceptron[0]
    w1 = perceptron[1]
    b  = perceptron[2]
    a = x0 * w0 + x1 * w1 + b
    return activation_function(a)

def compute_deltas(learning_rate, err, x0, x1, derivative_function):
    dw0 = learning_rate * err * derivative_function(err) * x0
    dw1 = learning_rate * err * derivative_function(err) * x1
    db = learning_rate * err * derivative_function(err) * 1

    return [dw0, dw1, db]

def back_propagate_errors(perceptron, deltas):
    for i in range(len(perceptron)):
        perceptron[i] += deltas[i]


def train_perceptron(data_set, no_of_epochs, learning_rate):
    w0 = random.random()
    w1 = random.random()
    b  = random.random()
    perceptron = [w0, w1, b]

    for i in range(no_of_epochs):
        for training_instance in data_set:
            x0, x1, y = training_instance
            o = feed_forward_info(x0, x1, perceptron, f)
            if o > 0.5:
                o = 1
            else:
                o = 0
            err = o - y
            deltas = compute_deltas(learning_rate, err, x0, x1, df)
            back_propagate_errors(perceptron, deltas)
            

    return perceptron

def test_perceptron(data_set, perceptron):
    TP, FP, TN, FN = 0, 0, 0, 0
    for testing_instance in data_set:
        x0, x1, y = testing_instance
        o = feed_forward_info(x0, x1, perceptron, f)

        if o > 0.5:
            o = 1
        else:
            o = 0

        print(x0, " ", x1, " ", y, " ", o)
        if int(y) == 1 and int(o) == 1:
            TP += 1
        if int(y) == 0 and int(o) == 0:
            TN += 1
        if int(y) == 0 and int(o) == 1:
            FP += 1
        if int(y) == 1 and int(o) == 0:
            FN += 1

    print(TP, " ", FP, " ", TN, " ", FN)
    
    return TP, FP, TN, FN


data_set = read_file('./modelAND.csv')

perceptron = train_perceptron(data_set, 100, 0.00001)

test_data_set = read_file('./testingDataSet.csv')

TP, FP, TN, FN = test_perceptron(test_data_set, perceptron)

print((TP + TN) / ( TP + FP + TN + FN ))
