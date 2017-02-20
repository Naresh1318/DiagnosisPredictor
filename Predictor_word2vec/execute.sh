#!/bin/bash

# Run this to execute logistic_regression, knn, random_forest, svm, decision_tree, dense_fully_connected and cnn.

# Run decision tree
echo "Training decision_tree"
python3 decision_tree.py > decision_tree.txt
echo "decision_tree trained!!"
echo "1/6"

# Run KNN
echo "Training knn"
python3 knn.py > knn.txt
echo "knn trained!!"
echo "2/6"

# Run logistic_regression
echo "Training logistic_regression"
python3 logistic_regression.py > logistic_regression.txt
echo "logistic_regression trained!!"
echo "3/6"

# Run random_forest
echo "Training random_forest"
python3 random_forest.py > random_forest.txt
echo "random_forest trained!!"
echo "4/6"


# Run dense_fully_connected
echo "Training dense_fully_connected"
python3 dense_fully_connected.py > dense_fully_connected.txt
echo "dense_fully_connected trained!!"
echo "5/6"

# Run CNN
echo "Training CNN"
python3 cnn.py > cnn.txt
echo "cnn trained!!"
echo "6/6"
