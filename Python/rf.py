#!/usr/bin/python2.7

import loading
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def _load_training_and_test_sets():
    """
    Loads all training and test number sets from file.
    Returns them along with appropriate labels.
    """
    class_labels = []
    test_labels = []

    for i in range(0, 10):
        [training, test] = loading.load_number_set(i)
        labels = [str(i)] * training.shape[0]
        tlabels = [str(i)] * test.shape[0]
        if i == 0:
            train_points = training
            test_points = test
        else:
            train_points = np.concatenate((train_points, training), axis = 0)
            test_points = np.concatenate((test_points, test), axis = 0)
        class_labels.extend(labels)
        test_labels.extend(tlabels)

    return train_points, test_points, class_labels, test_labels


def randomforest_identification(is_debug = False, trees=50):
    """
    Runs identification checks withing written number only using KNN method.
    During tests one-vs-one checks are run.
    Returns confusion matrixes for both training and test data sets.
    """
    conf_matrix = np.zeros((10, 10))
    conf_matrix2 = np.zeros((10, 10))
    train_points, test_points, class_labels, test_labels = _load_training_and_test_sets()

    # Random Forest routine
    rf = RandomForestClassifier(n_estimators=trees)
    rf.fit(train_points, class_labels)
    results = rf.predict(train_points)
    results2 = rf.predict(test_points)

    # Fill confusion matrixes
    for i in range(0, len(class_labels)):
        index = int(class_labels[i])
        conf_matrix[index, int(results[i])] += 1
    for i in range(0, len(test_labels)):
        index = int(test_labels[i])
        conf_matrix2[index, int(results2[i])] += 1

    if is_debug:
        print 'Error ratio ' + str(1.0 - vectors.score(train_points, class_labels))
        print 'Error ratio ' + str(1.0 - vectors.score(test_points, test_labels))

    return conf_matrix, conf_matrix2


def randomforest_classification(trees=50):
    conf_matrix = np.zeros((11, 11))
    conf_matrix2 = np.zeros((11, 11))
    rf_vectors = []

    # Read training and test sets
    for i in range(0, 10):
        [set1, t] = loading.load_number_set(i, 0.5)
        labels = [str(0)] * set1.shape[0]

        for j in range(0, 10):
            if j == i: continue
            [set2_part, t] = loading.load_number_set(j)
            if j == 0 or (j == 1 and i == 0):
                set2 = set2_part
            else:
                set2 = np.concatenate((set2, set2_part), axis = 0)

        labels2 = [str(1)] * set2.shape[0]
        labels.extend(labels2)
        rf_vectors.append(RandomForestClassifier(n_estimators=trees))
        rf_vectors[i].fit(np.concatenate((set1, set2), axis = 0), labels)

    for i in range(0, 10):
        [train, test] = loading.load_number_set(i)

        # Check training set
        for point in train:
            for j in range(0, 10):
                result = rf_vectors[j].predict(point.reshape(1, -1))
                if int(result[0]) == 0:
                    conf_matrix[i][j] += 1
                    break
                elif j == 9:
                    conf_matrix[i][10] += 1

        # Check test set
        for point in test:
            for j in range(0, 10):
                result = rf_vectors[j].predict(point.reshape(1, -1))
                if int(result[0]) == 0:
                    conf_matrix2[i][j] += 1
                    break
                elif j == 9:
                    conf_matrix2[i][10] += 1

    # Last check - letters
    letters = loading.load_letter_set()
    for letter in letters:
        for j in range(0, 10):
            result = rf_vectors[j].predict(letter.reshape(1, -1))
            if int(result[0]) == 0:
                conf_matrix2[10][j] += 1
                break
            elif j == 9:
                conf_matrix2[10][10] += 1

    return conf_matrix, conf_matrix2
