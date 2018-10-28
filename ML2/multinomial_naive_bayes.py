import glob
import math
import sys
from collections import Counter

LABEL = ['ham', 'spam']

if __name__ == '__main__':
    train_folder = str(sys.argv[1])
    test_folder = str(sys.argv[2])
    ham_train_path = train_folder+'/ham/*.txt'
    spam_train_path = train_folder+'/spam/*.txt'
    ham_test_path = test_folder+'/ham/*.txt'
    spam_test_path = test_folder+'/spam/*.txt'
    prior = dict()
    likelihood = dict()


    def pre_process_data(file_data, clas, train):
        counter = Counter(file_data)
        for word in train:
            likelihood[word + clas] = float(counter.get(word,0) + 1) / (len(file_data) + len(train))


    def read_train_data(path):
        files = glob.glob(path)
        words = list()
        document_count = 0
        for file in files:
            f = open(file, 'r', encoding="ISO-8859-1")
            file_word = f.read().split()
            document_count += 1
            words.extend(file_word)
        return words, document_count


    # Spam and Ham Train Data
    train_ham, train_ham_count = read_train_data(ham_train_path)
    train_spam, train_spam_count = read_train_data(spam_train_path)

    # All words in training set
    train = train_ham + train_spam

    # Distinct number of words in training set
    train_distinct = list(set(train))

    # Prior Probability
    prior[LABEL[1]] = float(train_spam_count) / (train_ham_count + train_spam_count)
    prior[LABEL[0]] = float(train_ham_count) / (train_ham_count + train_spam_count)

    pre_process_data(train_ham, LABEL[0], train_distinct)
    pre_process_data(train_spam, LABEL[1], train_distinct)


    def check_test_data(path, given_class):
        files = glob.glob(path)
        right_prediction = 0
        wrong_prediction = 0
        for file in files:
            f = open(file, 'r', encoding="ISO-8859-1")
            words = f.read().split()
            words_counter = Counter(words)
            max_score = -10000000.0
            predicted_class = None
            for clas in LABEL:
                score = math.log(prior[clas])
                for word, count in words_counter.items():
                    score += count * (math.log(likelihood.get(word + clas, 1)))
                if score > max_score:
                    max_score = score
                    predicted_class = clas
            if predicted_class == given_class:
                right_prediction += 1
            else:
                wrong_prediction += 1
        return right_prediction, wrong_prediction


    ham_right_prediction, ham_wrong_prediction = check_test_data(ham_test_path, LABEL[0])
    print("Accuracy ham file:" + str((ham_right_prediction / (ham_right_prediction + ham_wrong_prediction)) * 100))

    spam_right_prediction, spam_wrong_pediction = check_test_data(spam_test_path, LABEL[1])
    print("Accuracy spam file:" + str((spam_right_prediction / (spam_right_prediction + spam_wrong_pediction)) * 100))
    accuracy = float(ham_right_prediction + spam_right_prediction) / (
                ham_right_prediction + ham_wrong_prediction + spam_right_prediction + spam_wrong_pediction)
    print("Accuracy: "+str(accuracy*100))


