import argparse
import numpy as np

node = namedtuple("node", "fVal, nPosNeg, gain, left, right")


def parse_data_file_args():
    args_labels = ["-train_data", "-train_label", "-test_data", "-nLevels", "-pThrd", "-impurity", "-pred_file"]
    parser = argparse.ArgumentParser()

    for arg in args_labels:
        parser.add_argument(arg)

    return parser.parse_args()

def preprocess_train_data(args):
    training_set = np.genfromtxt(str(args.train_data), delimiter=' ')
    training_labels = np.genfromtxt(str(args.train_label), delimiter=' ')
    test_set = np.genfromtxt(str(args.test_data), delimiter=' ')

def main():
    args = parse_data_file_args()
    preprocess_train_data(args)

if __name__ == "__main__":
    main()
