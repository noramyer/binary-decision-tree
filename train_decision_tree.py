import argparse
import numpy as np
from collections import namedtuple
from node_class import Node

args = ""

def parse_data_file_args():
    global args

    args_labels = ["-train_data", "-train_label", "-test_data", "-test_label", "-nlevels", "-pthrd", "-impurity", "-pred_file"]
    parser = argparse.ArgumentParser()

    for arg in args_labels:
        parser.add_argument(arg)

    args = parser.parse_args()

def preprocess_train_data():
    training_set = np.genfromtxt(str(args.train_data), delimiter=' ')
    training_labels = np.genfromtxt(str(args.train_label), delimiter=' ')
    test_set = np.genfromtxt(str(args.test_data), delimiter=' ')
    test_labels = np.genfromtxt(str(args.test_label), delimiter=' ')

    return training_set, training_labels, test_set, test_labels

def classify(test_data, output_file, dt):
    test_data_classifications = []
    f = open(output_file, "w+")
    for d in test_data:
        label = dt.classify(d)
        f.write(str(label) + "\n")

        test_data_classifications.append(label)

def accuracy(test_data_classifications, test_data_true_labels):
    predicted = np.array(test_data_classifications)
    true_values = np.array(test_data_true_labels)

    return float(np.sum(predicted == true_values)) / len(test_data_classifications)

def main():
    parse_data_file_args()
    training_set, training_labels, test_set, test_labels = preprocess_train_data()
    n = Node(training_set, training_labels, str(args.impurity), str(args.nlevels))
    dt = n.build_decision_tree(training_set, training_labels, str(args.impurity), str(args.nlevels), str(args.pthrd))

if __name__ == "__main__":
    main()
