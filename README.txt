Author: Nora Myer.41

Requirements and assumptions for running:
  - This program can be run on Mac or linux (stdlinux) environments
  - Python version used: 2.7.10 or greater
    - Works with python3 as well
  - Make sure numpy is installed
  - Test data classifications printed to file_output.txt
  - Classification accuracy printed to console
  - Files written include node_class.py and train_decision_tree.py

File descriptions:
  - node_class.py is a python class representing a node which is used to build decision tree
  - train_decision_tree.py contains main() and:
      - reads in input
      - importing node class
      - using it to receive dt and then classify test data

Command to run:
$ python train_decision_tree.py -train_data train_data.txt -train_label train_label.txt -test_data test_data.txt -test_label test_label.txt -nlevels 40 -pthrd 0.0 -impurity gini -pred file_output.txt
