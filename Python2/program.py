#!/usr/bin/python2.7
import numpy as np
import svm

if __name__ == "__main__":
    m1, m2 = svm.get_identification1_results(gamma=0.5, should_normalize=True)
    np.savetxt("Results/SVM_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
    m1, m2 = svm.get_identification2_results(gamma=0.5, should_normalize=True)
    np.savetxt("Results/SVM_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
