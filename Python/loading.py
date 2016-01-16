#!/usr/bin/python2.7

import csv
import numpy
# from sklearn.preprocessing import normalize

def _normalize_matrix(matrix):
    row_sums = matrix.sum(axis = 1)
    return matrix / row_sums[:, numpy.newaxis]
#    return normalize(matrix, norm='l1', axis = 1)


def load_number_set(number_to_load, division_ratio = 0.5):
    """Loads and returns two sets from number set"""
    """Those sets are divided using division_ratio argument"""

    data = list()
    reader = csv.reader(open('dane.csv', 'r'), delimiter=',')
    for row in reader:
        if int(row[0]) == int(number_to_load):
            del row[0]
            data.append(row)

    matrix = numpy.array(data).astype('double')
    #    number_of_elements = normalized_matrix.shape[1]
    #    half = int(number_of_elements * division_ratio)
    return numpy.array_split(matrix, 2)


def load_letter_set(letter_to_load = '0'):
    data = list()
    reader = csv.reader(open('letters.csv', 'r'), delimiter=',')
    for row in reader:
        if str(letter_to_load) == '0' or str(row[0]) == str(letter_to_load):
            del row[0]
            data.append(row)

    return numpy.array(data).astype('double')
