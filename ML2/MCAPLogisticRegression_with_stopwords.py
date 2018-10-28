import glob
import math
import sys
import time
from itertools import chain

import numpy as np

from stopwords import stopwords

if __name__ == '__main__':
    train_folder = str(sys.argv[1])
    test_folder = str(sys.argv[2])
    ham_train_path = train_folder + '/ham/*.txt'
    spam_train_path = train_folder + '/spam/*.txt'
    ham_test_path = test_folder + '/ham/*.txt'
    spam_test_path = test_folder + '/spam/*.txt'

    regularized_parameter = 2
    learning_rate = 0.01
    iteration = 100


    def read_data(file):
        data = open(file, 'r', encoding="ISO-8859-1")
        data = data.read().split()
        data = [d for d in data if d not in stopwords]
        return data


    def create_read_matrix(file_name):
        files = glob.glob(file_name)
        return [read_data(file) for file in files], len(files)


    # flatten the 2D list into 1D list
    def flatten_matrix(matrix):
        return list(chain.from_iterable(matrix))


    def mutually_exclusive_features(spam_clean_list, ham_clean_list):
        result = list()
        result.extend(list(set(spam_clean_list).difference(ham_clean_list)))
        result.extend(list(set(ham_clean_list).difference(result)))
        return result


    def matrix_creation(matrix, feature_matrix, mef):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                feature_matrix[i][mef.index(matrix[i][j])] = matrix[i].count(matrix[i][j])


    def sigmoid(x):
        try:
            d = (1 + np.exp(-x))
            p = float(1) / d
            return p
        except Exception:
            return 0.0


    def sum_of_theta_and_feature_multiplication(file_count):
        for i in range(file_count):
            default_sum = 1.0
            for j in range(len(train_mef)):
                default_sum += theta[j] * features_matrix[i][j]
            probability = sigmoid(default_sum)
            sigmoid_vector[i] = probability


    def update_theta(theta_size):
        for i in range(theta_size):
            default_difference = theta0
            for doc_id in range(train_set_count):
                y = train_label[doc_id]
                h = sigmoid_vector[doc_id]
                default_difference += features_matrix[doc_id][i] * (y - h)
            prev_theta = theta[i]
            theta[i] += learning_rate * (default_difference - (regularized_parameter * prev_theta))


    # read data for wach file in train and test folder and return the matrix and count of files
    train_spam, train_spam_count = create_read_matrix(spam_train_path)
    train_ham, train_ham_count = create_read_matrix(ham_train_path)
    # Import variable for matrix manipulation
    train_set_count = train_spam_count + train_ham_count
    total_train = train_spam + train_ham
    train_spam_list = flatten_matrix(train_spam)
    train_ham_list = flatten_matrix(train_ham)
    train_mef = mutually_exclusive_features(train_spam_list, train_ham_list)
    features_matrix = np.zeros((train_set_count, len(train_mef)))
    theta = [0.0] * len(train_mef)
    theta0 = 0
    spam_value = [1] * train_spam_count
    ham_value = [0] * train_ham_count
    train_label = spam_value + ham_value

    matrix_creation(total_train, features_matrix, train_mef)
    sigmoid_vector = [0.0] * train_set_count


    def train():
        for i in range(iteration):
            sum_of_theta_and_feature_multiplication(train_set_count)
            update_theta(len(theta))


    test_spam, test_spam_count = create_read_matrix(spam_test_path)
    test_ham, test_ham_count = create_read_matrix(ham_test_path)
    test_set_count = test_ham_count + test_spam_count
    total_test = test_spam + test_ham
    test_spam_list = flatten_matrix(test_spam)
    test_ham_list = flatten_matrix(test_ham)
    test_spam_value = [1] * test_spam_count
    test_ham_value = [0] * test_ham_count
    test_label = test_spam_value + test_ham_value

    test_mef = mutually_exclusive_features(test_spam_list, test_ham_list)
    test_features_matrix = np.zeros((test_set_count, len(test_mef)))
    matrix_creation(total_test, test_features_matrix, test_mef)


    def test():
        ham_right_prediction = 0
        ham_wrong_prediction = 0
        spam_right_prediction = 0
        spam_wrong_prediction = 0

        for doc_id in range(test_set_count):
            sum = 1.0
            for i in range(len(test_mef)):
                word = test_mef[i]
                if word in train_mef:
                    sum += theta[train_mef.index(word)] * test_features_matrix[doc_id][i]
            h = sigmoid(sum)
            if h > 0.5:
                if test_label[doc_id] == 1:
                    spam_right_prediction += 1
                else:
                    spam_wrong_prediction += 1
            else:
                if test_label[doc_id] == 0:
                    ham_right_prediction += 1
                else:
                    ham_wrong_prediction += 1

        print("Accuracy ham file:" + str((ham_right_prediction / (ham_right_prediction + ham_wrong_prediction)) * 100))
        print(
            "Accuracy spam file:" + str((spam_right_prediction / (spam_right_prediction + spam_wrong_prediction)) * 100))
        print("Accuracy :" + str(((spam_right_prediction + ham_right_prediction) / (
                spam_right_prediction + spam_wrong_prediction + ham_right_prediction + ham_wrong_prediction)) * 100))

    train()
    test()
