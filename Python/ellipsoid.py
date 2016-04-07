#!/usr/bin/python2.7

import loading
import numpy as np
import numpy.linalg as la

pi = np.pi
sin = np.sin
cos = np.cos

def _mvee(points, tol = 0.001):
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d
    return np.asarray(A), np.squeeze(np.asarray(c))


def _get_ellipsoid_error(A, c, points, tolerance):
    counter = 0.0
    for row in points:
        point = np.asmatrix(row) - c
        dist = point * A * np.transpose(point)
        if dist > 1. + float(tolerance):
            counter += 1
    return counter / points.shape[0]


def _get_best_ellipsoid_number_and_error(A_list, c_list, row):
    point = np.asmatrix(row) - c_list[0]
    min_dist = point * A_list[0] * np.transpose(point)
    best_ellipse = 0

    for i in range(1, 10):
        point = np.asmatrix(row) - c_list[i]
        dist = point * A_list[i] * np.transpose(point)
        if float(dist) < float(min_dist):
            min_dist = dist
            best_ellipse = i

    return best_ellipse, float(min_dist)


def ellipsoids_letters_vs_numbers(tolerance, accuracy = 0.01, normalize=False):
    """
        Tries to classify and identify provided symbols
        and returns error ratio confusion matrix.
    """
    train_ratios = np.zeros((11, 11))
    test_ratios = np.zeros((11, 11))
    A_list = []
    c_list = []
    train_points = []
    test_points = []
    norm = None
    if normalize == True:
        norm = loading.get_normalize_vector()

    # Creating ellipsoids
    for i in range(0, 10):
        [train, test] = loading.load_number_set(i, 0.7, norm_vector=norm)
        train_points.append(train)
        test_points.append(test)
        A, c = _mvee(train, accuracy)
        A_list.append(A)
        c_list.append(c)

    # Append letters
    train_points.append(loading.load_letter_set(norm_vector=norm))
    test_points.append(loading.load_letter_set(norm_vector=norm))

    # Get error ratio for train set
    for i in range(0, 11):
        for row in train_points[i]:
            best_ellipse, dist = _get_best_ellipsoid_number_and_error(A_list, c_list, row)
            if dist > 1. + float(tolerance):
                best_ellipse = 10
            train_ratios[i, best_ellipse] += 1
    # Get error ratio for test set
    for i in range(0, 11):
        for row in test_points[i]:
            best_ellipse, dist = _get_best_ellipsoid_number_and_error(A_list, c_list, row)
            if dist > 1. + float(tolerance):
                best_ellipse = 10
            test_ratios[i, best_ellipse] += 1

    return train_ratios, test_ratios


def getIdentifiedPoints(normalize=False, useTrain=False):
    accuracy = 0.1
    tolerance = 0.5
    A_list = []
    c_list = []
    point_labels = []
    all_points = []
    norm = None
    if normalize == True:
        norm = loading.get_normalize_vector()

    # Creating ellipsoids
    for i in range(0, 10):
        [train, _] = loading.load_number_set(i, 0.7, norm_vector=norm)
        A, c = _mvee(train, accuracy)
        A_list.append(A)
        c_list.append(c)

    for i in range(0, 11):
        if i != 10:
            if useTrain:
                [train, _] = loading.load_number_set(i, 0.7, norm_vector=norm)
            else:
                [_, train] = loading.load_number_set(i, 0.7, norm_vector=norm)
        else:
            train = loading.load_letter_set(norm_vector=norm)
        for row in train:
            _, dist = _get_best_ellipsoid_number_and_error(A_list, c_list, row)
            if dist < 1. + float(tolerance):
                all_points.append(row)
                point_labels.append(i)

    return all_points, point_labels


def final_ellipsoids(data, primal_labels, new_labels, normalize=False):
    matrix = np.zeros((11, 11))
    A_list = []
    c_list = []
    accuracy = 0.1
    tolerance = 0.5
    norm = None
    if normalize == True:
        norm = loading.get_normalize_vector()

    # Prepare ellipsoids
    for i in range(0, 10):
        [train, _] = loading.load_number_set(i, 0.7, norm_vector=norm)
        A, c = _mvee(train, accuracy)
        A_list.append(A)
        c_list.append(c)

    # Check point alliance
    for i in range(0, len(new_labels)):
        label = int(new_labels[i])
        A = A_list[label]
        c = c_list[label]
        point = np.asmatrix(data[i, :]) - c
        distance = point * A * np.transpose(point)
        if float(distance) >= 1. + float(tolerance):
            label = 10
        matrix[int(primal_labels[i])][label] += 1

    return matrix
