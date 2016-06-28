#!/usr/bin/python2.7
import numpy as np
import classifier as c
import getopt, sys
import multiprocessing as mp

def _identification2_function1(biggest, second_biggest):
    return biggest > second_biggest + 1


def _identification2_function2(biggest, second_biggest):
    return biggest > 8


def _start_svm():
    print "Starting SVM identification 1"
    m1, m2 = c.get_identification1_results("svm", should_normalize=True, progress=progress)
    np.savetxt("Results/SVM_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting SVM identification 2"
    m1, m2 = c.get_identification2_results("svm", _identification2_function1, should_normalize=True, progress=progress)
    np.savetxt("Results/SVM_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting SVM identification 2 ver. 2"
    m1, m2 = c.get_identification2_results("svm", _identification2_function2, should_normalize=True, progress=progress)
    np.savetxt("Results/SVM_identification_one_vs_one_version2.txt", np.concatenate((m1, m2), axis=0), delimiter=',')


def _start_rf():
    print "Starting RF identification 1"
    m1, m2 = c.get_identification1_results("rf", should_normalize=True, progress=progress)
    np.savetxt("Results/RF_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting RF identification 2"
    m1, m2 = c.get_identification2_results("rf", _identification2_function1, should_normalize=True, progress=progress)
    np.savetxt("Results/RF_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting RF identification 2 ver. 2"
    m1, m2 = c.get_identification2_results("rf", _identification2_function2, should_normalize=True, progress=progress)
    np.savetxt("Results/RF_identification_one_vs_one_version2.txt", np.concatenate((m1, m2), axis=0), delimiter=',')


def _start_knn():
    print "Starting KNN identification 1"
    m1, m2 = c.get_identification1_results("knn", should_normalize=True, progress=progress)
    np.savetxt("Results/KNN_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting KNN identification 2"
    m1, m2 = c.get_identification2_results("knn", _identification2_function1, should_normalize=True, progress=progress)
    np.savetxt("Results/KNN_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting KNN identification 2 ver. 2"
    m1, m2 = c.get_identification2_results("knn", _identification2_function2, should_normalize=True, progress=progress)
    np.savetxt("Results/KNN_identification_one_vs_one_version2.txt", np.concatenate((m1, m2), axis=0), delimiter=',')


def _start_linear_regression():
    print "Starting linear regression identification 1"
    m1, m2 = c.get_identification1_results("lr", should_normalize=True, progress=progress)
    np.savetxt("Results/LR_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting linear regression identification 2"
    m1, m2 = c.get_identification2_results("lr", _identification2_function1, should_normalize=True, progress=progress)
    np.savetxt("Results/LR_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting linear regression identification 2 ver. 2"
    m1, m2 = c.get_identification2_results("lr", _identification2_function2, should_normalize=True, progress=progress)
    np.savetxt("Results/LR_identification_one_vs_one_version2.txt", np.concatenate((m1, m2), axis=0), delimiter=',')


def _start_logistic_regression():
    print "Starting logistic regression identification 1"
    m1, m2 = c.get_identification1_results("llr", should_normalize=True, progress=progress)
    np.savetxt("Results/LLR_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting logistic regression identification 2"
    m1, m2 = c.get_identification2_results("llr", _identification2_function1, should_normalize=True, progress=progress)
    np.savetxt("Results/LLR_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')

    print "Starting logistic regression identification 2 ver. 2"
    m1, m2 = c.get_identification2_results("llr", _identification2_function2, should_normalize=True, progress=progress)
    np.savetxt("Results/LLR_identification_one_vs_one_version2.txt", np.concatenate((m1, m2), axis=0), delimiter=',')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "skrlzp", [])
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    progress = False
    for o, a in opts:
        if o == "-p":
            progress = True

    pool = mp.Pool()
    for o, a in opts:
        if o == "-s":
            pool.apply_async(_start_svm)
        elif o == "-r":
            pool.apply_async(_start_rf)
        elif o == "-k":
            pool.apply_async(_start_knn)
        elif o == "-l":
            pool.apply_async(_start_linear_regression)
        elif o == "-z":
            pool.apply_async(_start_logistic_regression)
    pool.close()
    pool.join()
