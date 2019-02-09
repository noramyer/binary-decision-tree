import argparse
import numpy as np
from collections import namedtuple

args = ""

def parse_data_file_args():
    global args

    args_labels = ["-train_data", "-train_label", "-test_data", "-nLevels", "-pThrd", "-impurity", "-pred_file"]
    parser = argparse.ArgumentParser()

    for arg in args_labels:
        parser.add_argument(arg)

    args = parser.parse_args()

def preprocess_train_data():
    training_set = np.genfromtxt(str(args.train_data), delimiter=' ')
    training_labels = np.genfromtxt(str(args.train_label), delimiter=' ')
    test_set = np.genfromtxt(str(args.test_data), delimiter=' ')

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
    preprocess_train_data()

if __name__ == "__main__":
    main()
