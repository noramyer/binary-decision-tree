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

def main():
    parse_data_file_args()
    preprocess_train_data()

if __name__ == "__main__":
    main()
