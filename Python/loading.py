#!/usr/bin/python2.7

import csv
import numpy


def get_normalize_vector():
    """
    Returns vector used in normalization that is extracted
    from all numbers in data set.
    """
    for i in range(0, 10):
        [train, test] = _load_number_set(i)
        if i == 0:
            all_numbers = numpy.concatenate((train, test), axis = 0)
        else:
            all_numbers = numpy.concatenate((all_numbers, train, test), axis = 0)
    normalized_vector =  numpy.amax(all_numbers, axis = 0)
    normalized_vector[normalized_vector == 0] = 0.001 # Changes all occurences of zero to small value
    return normalized_vector


def _load_number_set(number_to_load, division_ratio = 0.5):
    """
    Loads and returns two sets from number set
    Those sets are divided using division_ratio argument
    """
    data = list()
    reader = csv.reader(open('dane.csv', 'r'), delimiter=',')
    for row in reader:
        if int(row[0]) == int(number_to_load):
            del row[0]
            data.append(row)

    matrix = numpy.array(data).astype('double')
    number_of_elements = matrix.shape[0]
    half = int(number_of_elements * division_ratio)
    train = matrix[:half, :]
    test = matrix[half:, :]
    return [train, test]


def load_number_set(number_to_load, division_ratio = 0.5, norm_vector = None):
    """
    Returns two sets with normalized data. For parameters explanation
    look at load_number_set function.
    """
    [train, test] = _load_number_set(number_to_load, division_ratio)
    if norm_vector is not None:
        train = train / norm_vector[None, :]
        test = test / norm_vector[None, :]
    return [train, test]


def _load_letter_set(set_to_load = 0, letter_to_load = '0'):
    """
    Loads certain letters from letters set and returns them.
    If letter_to_load is '0' then all letters will be loaded.
    If set_to_load is 0 then the letters will be combined from
    all two alternative letter sets.
    """
    data = list()
    set_filenames = { 0: 'letters.csv', 1 : 'letter_set_1.csv', 2 : 'letter_set_2.csv' }
    reader = csv.reader(open(set_filenames[set_to_load], 'r'), delimiter=',')
    for row in reader:
        if str(letter_to_load) == '0' or str(row[0]) == str(letter_to_load):
            del row[0]
            data.append(row)

    return numpy.array(data).astype('double')

def load_letter_set(set_to_load = 0, letter_to_load = '0', norm_vector = None):
    """
    Loads selected letter from certain letters set. See load_letter_set
    method for some parameters explanation. This matrix is also normalized
    using provided vector of maximal column values.
    norm_vector should be numpy vertical vector holding maximum values for
    each column.
    """
    normalized_matrix = _load_letter_set(set_to_load, letter_to_load)
    if norm_vector is not None:
        normalized_matrix = normalized_matrix / norm_vector[None, :]
    return normalized_matrix
