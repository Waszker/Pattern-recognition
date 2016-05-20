#!/usr/bin/python2.7

import loading
import numpy as np
from sklearn import svm

def _get_number_sets(should_normalize=False):
    """
    Returns lists containing training points, test points and their corresponding labels.
    """
    norm_vector = None
    if should_normalize:
        norm_vector = loading.get_normalize_vector()
    train_points = []
    test_points = []
    train_labels = []
    test_labels = []

    for i in range(0, 10):
        [train, test] = loading.load_number_set(i, 0.7, norm_vector)
        train_points.append(train)
        test_points.append(test)
        train_labels.append([str(i)] * train.shape[0])
        test_labels.append([str(i)] * test.shape[0])

    return train_points, test_points, train_labels, test_labels


def _get_one_versus_all_svms(c=8, kernel="rbf", gamma=0.125, should_normalize=False):
    """
    Prepares one - versus - others training sets which are used in svm training.
    After preparations all svms are returned in a vector.
    """
    # Preparation variables
    train_points, _, _, _ = _get_number_sets(should_normalize)

    # Now prepare one-vs-others svm
    svms = []
    s = svm.SVC(C=c, kernel=kernel, gamma=gamma)
    for i in range(0, 10):
        set1 = train_points[i]
        set2 = None
        for j in range(0, 10):
            if j == i: continue
            points = train_points[j]
            part = int(points.shape[0] * (1./9.))
            if set2 is None: set2 = points[:part, :]
            else: set2 = np.concatenate((set2, points[:part, :]), axis=0)

        # Having two sets prepared it's time to train svm
        labels = ["1"] * set1.shape[0]
        labels.extend(["2"] * set2.shape[0])
        svms.append(s.fit(np.concatenate((set1, set2), axis=0), labels))

    return svms


def _identify_and_classify_point(svms, point, conf_matrix, origin):
    prediction = 10
    for j in range(0, 10):
        # Check which class wants this sample
        result = svms[j].predict(point.reshape(1, -1))
        if result[0] == '1':
            prediction = j
            break
    # Save result
    conf_matrix[origin, prediction] += 1


def get_identification1_results(c=8, kernel="rbf", gamma=0.125, should_normalize=False):
    """
    Runs identification test for svm method trained with one-vs-others schema.
    If for every prediction "other" part identifies point, it's treated as foreign.
    """
    conf_matrix1 = np.zeros((11, 11), dtype=np.int)
    conf_matrix2 = np.zeros((11, 11), dtype=np.int)
    norm_vector = None
    if should_normalize:
        norm_vector = loading.get_normalize_vector()

    letters = loading.load_letter_set(norm_vector=norm_vector)
    svms = _get_one_versus_all_svms(c, kernel, gamma, should_normalize)
    train_points, test_points, _, _ = _get_number_sets(should_normalize)
    train_points.append(letters)
    test_points.append(letters)
    matrixes = [conf_matrix1, conf_matrix2]
    sets = [train_points, test_points]

    # Run tests on number sets
    for v in range(0, 2):
        conf_matrix = matrixes[v]
        point_set = sets[v]

        # Identify and classify points
        for i in range(0, len(point_set)):
            # i value represents original class
            points = point_set[i]
            for p in range(0, points.shape[0]):
                # Get signle point to classify
                point = points[p, :]
                _identify_and_classify_point(svms, point, conf_matrix, i)

    return conf_matrix1, conf_matrix2
