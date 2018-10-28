import copy
import random
import sys

import numpy as np
import pandas as pd


def variance_impurity_cal(target_column):
    type_of_values, counts = np.unique(target_column, return_counts=True)
    sum_of_counts = np.sum(counts)
    count_to_totals = [counts[i] / sum_of_counts for i in range(len(type_of_values))]
    variance = np.prod(count_to_totals)
    return variance


def variance_gain_cal(dataset, split_attribute_name, target_name="Class"):
    dataset_variance = variance_impurity_cal(dataset[target_name])
    type_of_values, counts = np.unique(dataset[split_attribute_name], return_counts=True)
    sum_of_counts = np.sum(counts)
    split_variance = [
        variance_impurity_cal(dataset.where(dataset[split_attribute_name] == type_of_values[i]).dropna()[target_name])
        for i in range(len(type_of_values))]
    split_variance_sum = np.sum([(counts[i] / sum_of_counts) * split_variance[i] for i in range(len(type_of_values))])
    variance_gain = dataset_variance - split_variance_sum
    return variance_gain


def entropy_cal(target_column):
    type_of_values, counts = np.unique(target_column, return_counts=True)
    sum_of_counts = np.sum(counts)
    count_to_totals = [counts[i] / sum_of_counts for i in range(len(type_of_values))]
    entropy = np.sum([-count_to_total * (np.log2(count_to_total)) for count_to_total in count_to_totals])
    return entropy


def information_gain_cal(dataset, split_attribute_name, target_name="Class"):
    dataset_entropy = entropy_cal(dataset[target_name])
    type_of_values, counts = np.unique(dataset[split_attribute_name], return_counts=True)
    sum_of_counts = np.sum(counts)
    split_entropy = [
        entropy_cal(dataset.where(dataset[split_attribute_name] == type_of_values[i]).dropna()[target_name]) for i in
        range(len(type_of_values))]
    split_entropy_sum = np.sum([(counts[i] / sum_of_counts) * split_entropy[i] for i in range(len(type_of_values))])
    information_gain = dataset_entropy - split_entropy_sum
    return information_gain


def decision_tree(partial_dataset, original_dataset, atrributes, type, target_name="Class", parent_class=None):
    if len(np.unique(partial_dataset[target_name])) <= 1:
        return np.unique(partial_dataset[target_name])[0], None
    elif len(partial_dataset) == 0:
        index_of_max_element = np.argmax(np.unique(original_dataset[target_name], return_counts=True)[1])
        return np.unique(original_dataset[target_name])[index_of_max_element], None
    # Need to check
    elif len(atrributes) == 0:
        return parent_class, None
    else:
        parent_class_index = np.argmax(np.unique(partial_dataset[target_name], return_counts=True)[1])
        parent_class = np.unique(partial_dataset[target_name])[parent_class_index]

        if type == 1:
            attributes_information_gain = [information_gain_cal(partial_dataset, atrribute, target_name) for atrribute
                                           in
                                           atrributes]
        else:
            attributes_information_gain = [variance_gain_cal(partial_dataset, atrribute, target_name) for atrribute in
                                           atrributes]
        most_gain_index = np.argmax(attributes_information_gain)
        best_attribute = atrributes[most_gain_index]

        root = dict()
        root[best_attribute] = dict()
        visualize_tree = dict()
        atrributes = [atrribute for atrribute in atrributes if atrribute is not best_attribute]

        for decision in np.unique(partial_dataset[best_attribute]):
            # print("Desicion : "+str(decision)+" "+str(best_attribute))
            sub_data = partial_dataset.where(partial_dataset[best_attribute] == decision).dropna()
            sub_tree = decision_tree(sub_data, original_dataset, atrributes, type, target_name, parent_class)
            root[best_attribute][decision] = sub_tree[0]
            visualize_tree_tuple = (best_attribute, decision)
            if isinstance(sub_tree[0], dict):
                visualize_tree[visualize_tree_tuple] = sub_tree[1]
            else:
                visualize_tree[visualize_tree_tuple] = sub_tree[0]
        return root, visualize_tree


def prediction(query, tree, default):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return prediction(query, result, default)
            else:
                return result


def get_non_leaf(tree, non_leaf_list):
    for key, value in tree.items():
        if isinstance(value, dict):
            if type(key) is str:
                non_leaf_list.append(key)
            get_non_leaf(value, non_leaf_list)
    return non_leaf_list


def update(tree, target_key, dataset):
    for key, value in tree.items():
        if key == target_key:
            prominent_index = np.argmax(np.unique(dataset[key], return_counts=True)[1])
            prominent_class = np.unique(dataset[key])[prominent_index]
            tree[key] = prominent_class
            break
        elif isinstance(value, dict):
            update(value, target_key, dataset)
    return tree


def post_pruning(L, K, tree, dataset, default):
    tree_best = copy.deepcopy(tree)
    tree_best_accuracy = test_data_set(dataset, tree_best, default)
    for i in range(L):
        dtree = copy.deepcopy(tree)
        m = random.randint(0, K)
        for j in range(m):
            non_leaf_list = list()
            non_leaf = get_non_leaf(dtree, non_leaf_list)
            if len(non_leaf):
                p = random.randint(0, len(non_leaf) - 1)
                target_element = non_leaf[p]
                dtree = update(dtree, target_element, dataset)
        if dtree:
            dtree_accuracy = test_data_set(dataset, dtree, default)
            if dtree_accuracy > tree_best_accuracy:
                tree_best = copy.deepcopy(dtree)
                tree_best_accuracy = dtree_accuracy
    return tree_best


def test_data_set(dataset, tree, default):
    test_dict = dataset.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predict"])
    for i in range(len(dataset)):
        predicted.loc[i, "predict"] = prediction(test_dict[i], tree, default)
    accuracy = (np.sum(predicted["predict"] == dataset["Class"]) / len(dataset)) * 100
    return accuracy


def visualize(tree, depth=0):
    if depth == 0:
        print('TREE Visualization \n')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        variable = str(split_criterion[0]).split('.')[0]
        value = str(split_criterion[1]).split('.')[0]
        if type(sub_trees) is dict:
            print('{0} = {1}:'.format(variable, value))
        else:
            sub_trees = str(sub_trees).split('.')[0]
            print('{0} = {1}:{2}'.format(variable, value, sub_trees))
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)


def load_data(file_name):
    df = pd.read_csv(file_name)
    return df


if __name__ == '__main__':
    L = int(int(sys.argv[1]))
    K = int(sys.argv[2])
    train_filename = str(sys.argv[3])
    validation_filename = str(sys.argv[4])
    test_filename = str(sys.argv[5])
    to_print = str(sys.argv[6])

    print("WITHOUT PRUNING USING INFORMATION GAIN\n")
    print("Loading Training Set for " + train_filename)
    df = load_data(train_filename)
    print("Training Set loaded\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, visualize_tree = decision_tree(df, df, df.columns[:-1], 1)
    print("Decision Tree created \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = load_data(validation_filename)
    print(('Validation Set loaded'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    accuracy = test_data_set(df_validation, tree, default)
    print('The prediction accuracy on Validation is: ', accuracy, '%')
    print('\n')
    print("Loading Test Set for " + test_filename)
    df_test = load_data(test_filename)
    print('Test Set loaded')
    default_index = np.argmax(np.unique(df_test['Class'], return_counts=True)[1])
    default = np.unique(df_test['Class'])[default_index]
    accuracy = test_data_set(df_test, tree, default)
    print('The prediction accuracy on Test Set is: ', accuracy, '%')
    print("\n\n\n")

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        visualize(visualize_tree)
    print("\n\n\n\n\n\n")

    print("WITHOUT PRUNING USING VARIANCE GAIN\n")
    print("Loading Training Set for " + train_filename)
    df = load_data(train_filename)
    print("Training Set loaded\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, visualize_tree = decision_tree(df, df, df.columns[:-1], 2)
    print("Decision Tree created \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = load_data(validation_filename)
    print(('Validation Set loaded'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    accuracy = test_data_set(df_validation, tree, default)
    print('The prediction accuracy on Validation is: ', accuracy, '%')
    print('\n')
    print("Loading Test Set for " + test_filename)
    df_test = load_data(test_filename)
    print('Test Set loaded')
    default_index = np.argmax(np.unique(df_test['Class'], return_counts=True)[1])
    default = np.unique(df_test['Class'])[default_index]
    accuracy = test_data_set(df_test, tree, default)
    print('The prediction accuracy on Test Set is: ', accuracy, '%')
    print("\n\n\n")

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        visualize(visualize_tree)
    print("\n\n\n\n\n\n")

    print("WITH PRUNING  USING INFORMATION GAIN\n")
    print('Using L=' + str(L) + " K=" + str(K))
    print("Loading Training Set for " + train_filename)
    df = load_data(train_filename)
    print("Training Set loaded\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, visualize_tree = decision_tree(df, df, df.columns[:-1], 1)
    print("Decision Tree created \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = load_data(validation_filename)
    print(('Validation Set loaded'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    print("Pruning Begins")
    tree = post_pruning(L, K, tree, df_validation, default)
    print('Pruning Completed')
    df_test = load_data(test_filename)
    default_index = np.argmax(np.unique(df_test['Class'], return_counts=True)[1])
    default = np.unique(df_test['Class'])[default_index]
    accuracy = test_data_set(df_test, tree, default)
    print('The prediction accuracy is: ', accuracy, '%')

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        visualize(visualize_tree)
    print("\n\n\n\n\n\n")

    print("WITH PRUNING  USING VARIANCE GAIN\n")
    print('Using L=' + str(L) + " K=" + str(K))
    print("Loading Training Set for " + train_filename)
    df = load_data(train_filename)
    print("Training Set loaded\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, visualize_tree = decision_tree(df, df, df.columns[:-1], 2)
    print("Decision Tree created \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = load_data(validation_filename)
    print(('Validation Set loaded'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    print("Pruning Begins")
    tree = post_pruning(L, K, tree, df_validation, default)
    print('Pruning Completed')
    df_test = load_data(test_filename)
    default_index = np.argmax(np.unique(df_test['Class'], return_counts=True)[1])
    default = np.unique(df_test['Class'])[default_index]
    accuracy = test_data_set(df_test, tree, default)
    print('The prediction accuracy is: ', accuracy, '%')

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        visualize(visualize_tree)
    print("\n\n\n\n\n\n")
