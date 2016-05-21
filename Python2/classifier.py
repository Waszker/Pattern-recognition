#!/usr/bin/python2.7

import sys
import loading
import progress_bar as pb
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def _get_number_sets(should_normalize=True):
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


def _get_svm(parameters):
    if parameters is None:
        parameters = {
            'C' : 8,
            'kernel' : 'rbf',
            'gamma' : 0.5
        }
    return svm.SVC(**parameters)


def _get_rf(parameters):
    if parameters is None:
        parameters = {
            'n_estimators' : 100,
        }
    return RandomForestClassifier(**parameters)


def _get_knn(parameters):
    if parameters is None:
        parameters = {
            'n_neighbours' : 5,
        }
    return KNeighborsClassifier(**parameters)


def _get_proper_classifier(classifier_name, parameters=None):
    try:
        return {
            'svm' : _get_svm(parameters),
            'rf' : _get_rf(parameters),
            'knn' : _get_knn(parameters),
        }[classifier_name]
    except KeyError:
        print "You must provide proper classifier name!"
        sys.exit(1)


def _get_one_versus_all_classifiers(classifier_name, parameters, should_normalize=True):
    """
    Prepares one - versus - others training sets which are used in training.
    After preparations all classifiers are returned in a list.
    """
    # Preparation variables
    train_points, _, _, _ = _get_number_sets(should_normalize)

    # Now prepare one-vs-others svm
    classifiers = []
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
        s = _get_proper_classifier(classifier_name, parameters)
        s.fit(np.concatenate((set1, set2), axis=0), labels)
        classifiers.append(s)

    return classifiers


def _get_one_versus_one_classifiers(classifier_name, parameters, should_normalize=True):
    """
    Prepares one - versus - one training sets and trains classifier.
    All trained classifiers are returned in a list.
    """
    # Preparation variables
    train_points, _, _, _ = _get_number_sets(should_normalize)

    # Now prepare one-vs-one svm
    classifiers = []
    for i in range(0, 10):
        set1 = train_points[i]
        for j in range(i+1, 10):
            set2 = train_points[j]

            # Having two sets prepared it's time to train svm
            labels = ["1"] * set1.shape[0]
            labels.extend(["2"] * set2.shape[0])
            s = _get_proper_classifier(classifier_name, parameters)
            s.fit(np.concatenate((set1, set2), axis=0), labels)
            classifiers.append(s)

    return classifiers


def _identify_and_classify_point(classifiers, point, conf_matrix, origin):
    prediction = 10
    for j in range(0, 10):
        # Check which class wants this sample
        result = classifiers[j].predict(point.reshape(1, -1))
        if result[0] == '1':
            prediction = j
            break
    # Save result
    conf_matrix[origin, prediction] += 1


def _get_needed_variables(should_normalize=True):
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


def _get_all_progress_info(sets_list):
    progress = 0

    for s in sets_list:
        for e in s:
            for p in e:
                progress += 1

    return progress


def get_identification1_results(classifier_name, parameters=None, should_normalize=True, progress=False):
    """
    Runs identification test for classifier method trained with one-vs-others schema.
    If for every prediction "other" part identifies point, it's treated as foreign.
    """
    classifiers = _get_one_versus_all_classifiers(classifier_name, parameters, should_normalize)
    matrixes, sets = _get_needed_variables(should_normalize)
    all_progress, current_progress = _get_all_progress_info(sets), 0

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
                current_progress += 1
                _identify_and_classify_point(classifiers, point, conf_matrix, i)
                if progress: pb.print_progress(current_progress, all_progress, prefix='Progress:', barLength=50)

    return matrixes


def _get_prediction_vector(classifiers, point):
    predictions = np.zeros((10))
    index = 0
    for i in range(0, 10):
        for j in range(i + 1, 10):
            result = classifiers[index].predict(point.reshape(1, -1))
            index += 1
            if result[0] == '1':
                predictions[i] += 1
            else:
                predictions[j] += 1

    return predictions


def get_identification2_results(classifier_name, parameters=None, should_normalize=True, progress=False):
    """
    Runs identification test for classifier method trained with one-vs-others schema.
    If for every prediction "other" part identifies point, it's treated as foreign.
    """
    classifiers = _get_one_versus_one_classifiers(classifier_name, parameters, should_normalize)
    matrixes, sets = _get_needed_variables(should_normalize)
    all_progress, current_progress = _get_all_progress_info(sets), 0

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
                current_progress += 1
                predictions = _get_prediction_vector(classifiers, point)
                if progress: pb.print_progress(current_progress, all_progress, prefix='Progress:', barLength=50)
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
