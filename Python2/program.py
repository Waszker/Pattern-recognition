#!/usr/bin/python2.7
import numpy as np
import classifier as c
import getopt, sys

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "skrp", [])
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    progress = False
    for o, a in opts:
        if o == "-p":
            progress = True

    for o, a in opts:
        if o == "-s":
            print "Starting SVM identification 1"
            m1, m2 = c.get_identification1_results("svm", should_normalize=True, progress=progress)
            np.savetxt("Results/SVM_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
            print "Starting SVM identification 2"
            m1, m2 = c.get_identification2_results("svm", should_normalize=True, progress=progress)
            np.savetxt("Results/SVM_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
        elif o == "-r":
            print "Starting RF identification 1"
            m1, m2 = c.get_identification1_results("rf", should_normalize=True, progress=progress)
            np.savetxt("Results/RF_identification_one_vs_all.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
            print "Starting RF identification 2"
            m1, m2 = c.get_identification2_results("rf", should_normalize=True, progress=progress)
            np.savetxt("Results/RF_identification_one_vs_one.txt", np.concatenate((m1, m2), axis=0), delimiter=',')
