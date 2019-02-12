import argparse
import numpy as np
from collections import namedtuple
from node_class import Node

#Author: Nora Myer
#Date: 2/5/19
#Description: This is main class that will use the node class to read in the data, build the decision_tree
#and then calc the classification accuracy of the decision_tree

args = ""

#Sets up and reads in args based in from the command line
def parse_data_file_args():
    global args

    #set arg labels
    args_labels = ["-train_data", "-train_label", "-test_data", "-test_label", "-nlevels", "-pthrd", "-impurity", "-pred_file"]
    parser = argparse.ArgumentParser()

    #build the parser with labels
    for arg in args_labels:
        parser.add_argument(arg)

    #set global args
    args = parser.parse_args()

#Pull training and test data out of arg parser
def preprocess_train_data():
    training_set = np.genfromtxt(str(args.train_data), delimiter=' ')
    training_labels = np.genfromtxt(str(args.train_label), delimiter=' ')
    test_set = np.genfromtxt(str(args.test_data), delimiter=' ')
    test_labels = np.genfromtxt(str(args.test_label), delimiter=' ')

    return training_set, training_labels, test_set, test_labels

#Given test data, return array of classifications for each row in test
def classify(test_data, output_file, dt):
    test_data_classifications = []

    #opens file to write classifications
    f = open(output_file, "w+")

    #for each row, classify based on features and print to file
    for d in test_data:
        label = dt.classify(d)
        f.write(str(label) + "\n")
        test_data_classifications.append(label)

    return test_data_classifications

#given predictions and true test labels, calculates the accuracy of the classification
def accuracy(test_data_classifications, test_data_true_labels):
    predicted = np.array(test_data_classifications)
    true_values = np.array(test_data_true_labels)

    return float(np.sum(predicted == true_values)) / len(test_data_classifications)

#BONUS
#This calculates the accuracy, precision, and recall assuming you have two classifications class1 and not class1
def confusion_matrix(test_data_classifications, test_data_true_labels):
    #set confusion matrix cells equal to zero
    true_predicted_class_1 = 0
    false_predicted_class_1 = 0
    true_predicted_not_class_1 = 0
    false_predicted_not_class_1 = 0
    count_positive = 0

    #for each of the test points, look at how it was correctly or incorrectly classified
    for idx, val in enumerate(test_data_true_labels):
        if val == 1.0:
            count_positive += 1

            #true positive case
            if test_data_classifications[idx] == 1.0:
                true_predicted_class_1 += 1
            #false negative case
            else:
                false_predicted_not_class_1 += 1
        else:
            #false positive case
            if test_data_classifications[idx] == 1.0:
                false_predicted_class_1 += 1
            #true negative case
            else:
                true_predicted_not_class_1 += 1

    #calculate values
    accuracy = float(true_predicted_not_class_1 + true_predicted_class_1) / len(test_data_classifications)
    precision = float(true_predicted_class_1) / (true_predicted_class_1 + false_predicted_class_1)
    recall = float(true_predicted_class_1 / count_positive)

    #print results to output
    print("**** Bonus: Class 1 Binary Classification Results ****")
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: "+ str(recall))

def main():
    parse_data_file_args()
    training_set, training_labels, test_set, test_labels = preprocess_train_data()

    #set up and build dt from imported node class
    print("Building decision tree... \n")
    n = Node(training_set, training_labels, str(args.impurity), str(args.nlevels))
    dt = n.build_decision_tree(training_set, training_labels, str(args.impurity), int(str(args.nlevels)), float(str(args.pthrd)))

    #given test data, get classifications from dt
    print("Classifying test data... \n")
    test_data_classifications = classify(test_set, str(args.pred_file), dt)

    #calc and print accuracy of labels
    a = accuracy(test_data_classifications, test_labels)
    print("Overall classification accuracy: " + str(a))
    print("Overall error rate: " + str(1.0 - a) +"\n")

    #BONUS: class 1 binary classification calculation method call
    a = confusion_matrix(test_data_classifications, test_labels)


if __name__ == "__main__":
    main()
