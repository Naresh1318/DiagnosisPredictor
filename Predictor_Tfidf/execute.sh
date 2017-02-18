#!/bin/bash

# Run this to execute logistic_regression, knn, random_forest, svm and decision_tree

# Run decision tree
echo "Training decision_tree"
python3 decision_tree.py > decision_tree.txt
echo "decision_tree trained!!"
echo "1/5"

# Run KNN
echo "Training knn"
python3 knn.py > knn.txt
echo "knn trained!!"
echo "2/5"

# Run logistic_regression
echo "Training logistic_regression"
python3 logistic_regression.py > logistic_regression.txt
echo "logistic_regression trained!!"
echo "3/5"

# Run random_forest
echo "Training random_forest"
python3 random_forest.py > random_forest.txt
echo "random_forest trained!!"
echo "4/5"

# Run svm
#echo "Training svm"
#python3 svm.py > svm.txt
#echo "svm trained!!"
#echo "5/5"

# Run LSTM
echo "Training LSTM"
python3 lstm.py
echo "LSTM trained!!"
echo "5/5"
