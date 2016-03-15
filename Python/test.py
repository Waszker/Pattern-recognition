#!/usr/bin/python2.7

import ellipsoid as e
import svn as s
import knn as k
import rf as rf
import numpy as np
import getopt, sys
from math import pow
import multiprocessing as mp

def _ellipsoid_functions(tol, acc, normalize):
    print 'Starting iteration for tol=' + str(tol) + ' acc=' + str(acc)
    m1, m2 = e.ellipsoids_letters_vs_numbers(tol, acc, normalize=normalize)
    filename = "results/ellipsoid_results_tol=" +  str(tol) + "_acc=" + str(acc) + ".txt"
    np.savetxt(filename, np.concatenate((m1, m2), axis = 0), delimiter=',')


def _ellipsoid_tests(normalize = False):
    print '*** Starting ellipsoids ***'
    pool = mp.Pool()
    accuracies = [0.1, 0.01, 0.001, 0.0005]
    tolerances = [0.5, 0.2, 0.1, 0.01]

    for i in range(0, len(accuracies)):
        for j in range(0, len(tolerances)):
            pool.apply_async(_ellipsoid_functions, args = (tolerances[j], accuracies[i], normalize))
    pool.close()
    pool.join()

    print '*** Ended ellipsoids ***'


def _svm_functions(c, gamma, kernel, normalize):
    """
    Runs two functions one after another with provided parameters.
    First one is identification problem using SVM method and the second
    one is classification problem (identification is run there too).
    Results are saved to files with appropriate names.
    """
    print 'Starting identification2 iteration for c=' + str(c) + ' gamma=' + str(gamma)
    m1, m2 = s.svm_identification2(c=c, gamma=gamma, kernel=kernel, normalize=normalize)
    filename = "results_svn/" + str(kernel) + "_identification2_c=" + str(c) + '_gamma=' + str(gamma)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

    print 'Starting identification iteration for c=' + str(c) + ' gamma=' + str(gamma)
    m1, m2 = s.svm_identification(c=c, gamma=gamma, kernel=kernel, normalize=normalize)
    filename = "results_svn/" + str(kernel) + "_identification_c=" + str(c) + '_gamma=' + str(gamma)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

    print 'Starting classification iteration for c=' + str(c) + ' gamma=' + str(gamma)
    m1, m2 = s.svm_classification(c=c, gamma=gamma, kernel=kernel, normalize=normalize)
    filename = "results_svn/" + str(kernel) + "_classification_c=" + str(c) + '_gamma=' + str(gamma)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

def _svm_tests(normalize = False):
    print '*** Starting SVM ***'
    pool = mp.Pool()
    kernels = ['rbf', 'poly']
    c_list = [1, 2, 4, 8, 16]
    gammas = [pow(2, -1), pow(2, -2), pow(2, -3), 0.00025]

    for kernel in kernels:
        for i in range(0, len(c_list)):
            for j in range(0, len(gammas)):
                pool.apply_async(_svm_functions, args = (c_list[i], gammas[j], kernel, normalize))
    pool.close()
    pool.join()

    print '*** Ended SVM ***'

def _knn_functions(n, normalize):
    """
    Runs two functions one after another with provided parameters.
    First one is identification problem using KNN method and the second
    one is classification problem (identification is run there too).
    Results are saved to files with appropriate names.
    """
    print 'Starting identification iteration for n=' + str(n)
    m1, m2 = k.knn_identification(n=n, normalize=normalize)
    filename = "results_knn/" + "knn" + "_identification_n=" + str(n)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

    print 'Starting classification iteration for n=' + str(n)
    m1, m2 = k.knn_classification(n=n, normalize=normalize)
    filename = "results_knn/" + "knn" + "_classification_n=" + str(n)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

def _knn_tests(normalize = False):
    print '*** Starting KNN ***'
    pool = mp.Pool()
    n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in range(0, len(n_list)):
        pool.apply_async(_knn_functions, args = (n_list[i], normalize))
    pool.close()
    pool.join()


    print '*** Ended KNN ***'

def _randomforest_functions(trees, normalize):
    """
    Runs two functions one after another with provided parameters.
    First one is identification problem using Random Forest method and the second
    one is classification problem (identification is run there too).
    Results are saved to files with appropriate names.
    """
    print 'Starting identification iteration for n=' + str(trees)
    m1, m2 = rf.randomforest_identification(trees=trees, normalize=normalize)
    filename = "results_rf/" + "rf" + "_identification_n=" + str(trees)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

    print 'Starting classification iteration for n=' + str(trees)
    m1, m2 = rf.randomforest_classification(trees=trees, normalize=normalize)
    filename = "results_rf/" + "rf" + "_classification_n=" + str(trees)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

def _randomforest_tests(normalize = False):
    print '*** Starting Random Forest ***'
    pool = mp.Pool()
    #tree_list = [50, 60, 70, 80, 90, 100]
    tree_list = [85, 95, 140, 150]

    for i in range(0, len(tree_list)):
        pool.apply_async(_randomforest_functions, args = (tree_list[i], normalize))
    pool.close()
    pool.join()

    print '*** Ended Random Forest ***'


def _test_all(normalize):
    sett, labels = e.getIdentifiedPoints(normalize)
    results = s.svm_identify_points(sett, labels, gamma=0.125, normalize=normalize)
    np.savetxt('final_results/results_svm.txt', results, delimiter=',')
    results = k.classifyPoints(sett, labels, n=4, normalize=normalize)
    np.savetxt('final_results/results_knn.txt', results, delimiter=',')
    results = rf.classifyPoints(sett, labels, trees=100, normalize=normalize)
    np.savetxt('final_results/results_rf.txt', results, delimiter=',')
    points, old_labels, new_labels = s.svm_classify_points(gamma=0.125, normalize=normalize)
    results = e.final_ellipsoids(points, old_labels, new_labels, normalize=normalize)
    np.savetxt('final_results/results_svm2.txt', results, delimiter=',')


if __name__ == "__main__":
    normalize = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "eskr", ["normalize", "zusammen"])
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    for o, a in opts:
        if o == "--normalize":
            normalize = True

    for o, a in opts:
        if o == "-e":
            _ellipsoid_tests(normalize)
        elif o == "-s":
            _svm_tests(normalize)
        elif o == "-k":
            _knn_tests(normalize)
        elif o == "-r":
            _randomforest_tests(normalize)
        elif o == "--zusammen":
            _test_all(normalize)
