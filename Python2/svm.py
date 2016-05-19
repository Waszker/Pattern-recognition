#!/usr/bin/python2.7

import loading
import numpy as np
from sklearn import svm

def _get_one_versus_all_svms(c=8, kernel="rbf", gamma=0.125, should_normalize = False):
    """
    Prepares one - versus - others training sets which are used in svm training.
    After preparations all svms are returned in a vector.
    """
    # Preparation variables
    norm_vector = None
    if should_normalize:
        norm_vector = loading.get_normalize_vector()
    train_points = []

    # Load corresponding training points to a vector
    for i in range(0, 10):
        [training, _] = loading.load_number_set(i, 0.7, norm_vector)
        train_points.append(training)

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
