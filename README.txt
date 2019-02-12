Author: Nora Myer.41

Requirements and assumptions for running:
  - This program can be run on Mac or linux (stdlinux) environments
  - Run using python3 on stdlinux since earlier numpy packages arent installed that work with python 2.7
  - Test data classifications printed to file_output.txt
  - Classification accuracy printed to console
  - Files written include node_class.py and train_decision_tree.py
  - Experiment with diff -nlevels -pthrd -impurity values
  - The overall accuracy results of the test data for the dt are printed
  - Then, the bonus accuracy, recall, and precision are calculated and printed to console under the following header:
    - "**** Bonus: Class 1 Binary Classification Results ****"

File descriptions:
  - node_class.py is a python class representing a node which is used to build decision tree
  - train_decision_tree.py contains main() and:
      - reads in input
      - importing node class
      - using it to receive dt and then classify test data

Command to run:
$ python3 train_decision_tree.py -train_data train_data.txt -train_label train_label.txt -test_data test_data.txt -test_label test_label.txt -nlevels 40 -pthrd 0.0 -impurity gini -pred file_output.txt
