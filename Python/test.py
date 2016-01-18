#!/usr/bin/python2.7

import ellipsoid as e
import svn as s
import numpy as np
import getopt, sys
from math import pow
import multiprocessing as mp

def _ellipsoid_functions(tol, acc):
    print 'Starting iteration for tol=' + str(tol) + ' acc=' + str(acc)
    m1, m2 = e.ellipsoids_letters_vs_numbers(tol, acc)
    filename = "results/ellipsoid_results_tol=" +  str(tol) + "_acc=" + str(acc) + ".txt"
    np.savetxt(filename, np.concatenate((m1, m2), axis = 0), delimiter=',')


def _ellipsoid_tests():
    print '*** Starting ellipsoids ***'
    pool = mp.Pool()
    accuracies = [0.1, 0.01, 0.001, 0.0005]
    tolerances = [0.5, 0.2, 0.1, 0.01]

    for i in range(0, len(accuracies)):
        for j in range(0, len(tolerances)):
            pool.apply_async(_ellipsoid_functions, args = (tolerances[j], accuracies[i]))
    pool.close()
    pool.join()

    print '*** Ended ellipsoids ***'


def _svm_functions(c, gamma, kernel):
    """
    Runs two functions one after another with provided parameters.
    First one is identification problem using SVM method and the second
    one is classification problem (identification is run there too).
    Results are saved to files with appropriate names.
    """
    print 'Starting identification iteration for c=' + str(c) + ' gamma=' + str(gamma)
    m1, m2 = s.svm_identification(c=c, gamma=gamma, kernel=kernel)
    filename = "results_svn/" + str(kernel) + "_identification_c=" + str(c) + '_gamma=' + str(gamma)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')

    print 'Starting classification iteration for c=' + str(c) + ' gamma=' + str(gamma)
    m1, m2 = s.svm_classification(c=c, gamma=gamma, kernel=kernel)
    filename = "results_svn/" + str(kernel) + "_classification_c=" + str(c) + '_gamma=' + str(gamma)
    np.savetxt(filename, np.concatenate((m1, m2), axis=0), delimiter=',')


def _svm_tests():
    print '*** Starting SVM ***'
    pool = mp.Pool()
    kernels = ['rbf', 'poly']
    c_list = [1, 2, 4, 8, 16]
    gammas = [pow(2, -1), pow(2, -2), pow(2, -3), 0.00025]

    for kernel in kernels:
        for i in range(0, len(c_list)):
            for j in range(0, len(gammas)):
                pool.apply_async(_svm_functions, args = (c_list[i], gammas[j], kernel))
    pool.close()
    pool.join()

    print '*** Ended SVM ***'


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "es")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    for o, a in opts:
        if o == "-e":
            _ellipsoid_tests()
        elif o == "-s":
            _svm_tests()
