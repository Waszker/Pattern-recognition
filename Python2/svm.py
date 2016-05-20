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
    After preparations all svms are returned in a list.
    """
    # Preparation variables
    train_points, _, _, _ = _get_number_sets(should_normalize)

    # Now prepare one-vs-others svm
    svms = []
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
        s = svm.SVC(C=c, kernel=kernel, gamma=gamma)
        s.fit(np.concatenate((set1, set2), axis=0), labels)
        svms.append(s)

    return svms


def _get_one_versus_one_svms(c=8, kernel="rbf", gamma=0.125, should_normalize=False):
    """
    Prepares one - versus - one training sets and trains svm.
    All trained svms are returned in a list.
    """
    # Preparation variables
    train_points, _, _, _ = _get_number_sets(should_normalize)

    # Now prepare one-vs-one svm
    svms = []
    for i in range(0, 10):
        set1 = train_points[i]
        for j in range(i+1, 10):
            set2 = train_points[j]

            # Having two sets prepared it's time to train svm
            labels = ["1"] * set1.shape[0]
            labels.extend(["2"] * set2.shape[0])
            s = svm.SVC(C=c, kernel=kernel, gamma=gamma)
            s.fit(np.concatenate((set1, set2), axis=0), labels)
            svms.append(s)

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


def _get_needed_variables(should_normalize=False):
    conf_matrix1 = np.zeros((11, 11), dtype=np.int)
    conf_matrix2 = np.zeros((11, 11), dtype=np.int)
    norm_vector = None
    if should_normalize:
        norm_vector = loading.get_normalize_vector()

    letters = loading.load_letter_set(norm_vector=norm_vector)
    train_points, test_points, _, _ = _get_number_sets(should_normalize)
    train_points.append(letters)
    test_points.append(letters)
    matrixes = [conf_matrix1, conf_matrix2]
    sets = [train_points, test_points]

    return matrixes, sets


def get_identification1_results(c=8, kernel="rbf", gamma=0.125, should_normalize=False):
    """
    Runs identification test for svm method trained with one-vs-others schema.
    If for every prediction "other" part identifies point, it's treated as foreign.
    """
    svms = _get_one_versus_all_svms(c, kernel, gamma, should_normalize)
    matrixes, sets = _get_needed_variables(should_normalize)

    # Run tests on number sets
    for v in range(0, 2):
        conf_matrix = matrixes[v]
        point_set = sets[v]

        # Identify and classify points
        for i in range(0, len(point_set)):
            # i value represents original class
            points = point_set[i]
            for point in points:
                # Get signle point to classify
                _identify_and_classify_point(svms, point, conf_matrix, i)

    return matrixes


def _get_prediction_vector(svms, point):
    predictions = np.zeros((10))
    index = 0
    for i in range(0, 10):
        for j in range(i + 1, 10):
            result = svms[index].predict(point.reshape(1, -1))
            index += 1
            if result[0] == '1':
                predictions[i] += 1
            else:
                predictions[j] += 1

    return predictions



def get_identification2_results(c=8, kernel="rbf", gamma=0.125, should_normalize=False):
    """
    Runs identification test for svm method trained with one-vs-others schema.
    If for every prediction "other" part identifies point, it's treated as foreign.
    """
    svms = _get_one_versus_one_svms(c, kernel, gamma, should_normalize)
    matrixes, sets = _get_needed_variables(should_normalize)

    # Run tests on number sets
    for v in range(0, 2):
        conf_matrix = matrixes[v]
        point_set = sets[v]

        # Identify and classify points
        for i in range(0, len(point_set)):
            # i value represents original class
            points = point_set[i]
            for point in points:
                # classify single point
                predictions = _get_prediction_vector(svms, point)
                # Find biggest and second biggest results and best index
                best, big, sbig = 0, 0, 0
                for l in range(0, len(predictions)):
                    if big < predictions[l]:
                        best, big, sbig = l, predictions[l], big
                    elif sbig < predictions[l]: sbig = predictions[l]
                if big > sbig + 1:
                    conf_matrix[i, best] += 1
                else:
                    conf_matrix[i, 10] += 1

    return matrixes
