#!/bin/python

import csv
import math

FILENAME = "titanic-homework.csv"
DEFAULT_TAB = "|   "

TITANIC_AGE_LABEL = "Age"
TITANIC_AGE_DATA = {
    "young":    [-1, 20],
    "middle":   [20, 40],
    "old":      [40, 100]
}


def titanic_age_group(age):
    for group in TITANIC_AGE_DATA:
        bounds = TITANIC_AGE_DATA[group]
        if int(age) > bounds[0] and int(age) < bounds[1]:
            return group
    # return "unknown"


def load_data(filename):
    lists = []
    labels = []
    dict = {}
    with open(filename) as file:
        data = csv.reader(file)
        i = 0
        for row in data:
            if i == 0:
                labels = row
                for x in row:
                    lists.append([])
            else:
                for x in range(len(labels)):
                    if labels[x] == TITANIC_AGE_LABEL:
                        lists[x].append(titanic_age_group(row[x]))
                    else:
                        lists[x].append(row[x])
            i += 1
    for x in range(len(labels)):
        dict[labels[x]] = lists[x]
    # return the dictionary and the amount of items - 1 for label
    return (dict, i - 1)


def calc_set_entropy(data, res):
    size = len(data[res])
    vals = {}
    entropy = 0
    for val in data[res]:
        vals.setdefault(val, 0)
        vals[val] += 1
    for key in vals:
        prob = vals[key] / size
        entropy -= prob * math.log(prob, 2)
    return entropy


def calc_attrib(data, attr, res):
    # data dictionary, attribute label, result label
    # returns conditional entropy and intrinsic info for a selected attribute
    results = {}
    attr_count = {}
    count = len(data[attr])
    for i in range(len(data[attr])):
        attrib = data[attr][i]
        result = data[res][i]

        results.setdefault(attrib, {})
        results[attrib].setdefault(result, 0)
        results[attrib][result] += 1

        attr_count.setdefault(attrib, 0)
        attr_count[attrib] += 1

    entropies = {}
    intr_info = 0
    cond_entropy = 0
    for attrib in results:
        for result in results[attrib]:
            prob = results[attrib][result] / attr_count[attrib]
            entropies.setdefault(attrib, 0)
            entropies[attrib] -= prob * math.log(prob, 2)

        x = attr_count[attrib] / count
        intr_info -= x * math.log(x, 2)

    # conditional entropy
    for attr in entropies:
        cond_entropy += entropies[attr] * attr_count[attr] / count
        # print("Entropy for", attr, entropies[attr])
    return (cond_entropy, intr_info)


def check_known_attribs(i, data, known_attr: {}):
    for chk in known_attr:
        if data[chk][i] not in known_attr[chk]:
            return False
    return True


def check_inner_dict_empty(dict):
    for x in dict:
        if not bool(dict[x]):
            return True
    return False


def id3(data, res, order=[], known_attr={}):
    if not bool(order):
        return {}
    dict = {}
    attr = order[0]

    # unique values on currently checked attribute
    unique_vals = set(data[attr])

    for val in unique_vals:
        new_known_attr = known_attr.copy()
        new_known_attr.setdefault(attr, [])
        new_known_attr[attr].append(val)

        # checking if all known attributes return the same result class
        last = None
        are_identical = False
        for i in range(len(data[res])):
            # leave this element if it doesnt match already known attributes
            if not check_known_attribs(i, data, new_known_attr):
                continue

            result_val = data[res][i]

            if last is None:
                last = result_val
                are_identical = True
            elif result_val != last:
                last = result_val
                are_identical = False
                break

        # if all matching elements result in the same class we skip to the next value
        if (are_identical):
            dict[val] = {res: last}
            continue

        # otherwise we are going deeper
        next = id3(data, res, order[1:], new_known_attr)

        # check if the dict is empty
        if last is not None:
            # check if theres an empty dict inside
            if not bool(next) or check_inner_dict_empty(next):
                dict[val] = {res: last}
            else:
                dict[val] = next

    if len(dict) == 1:
        [key] = dict.keys()
        return dict[key]
    else:
        return {attr: dict}


def printTree(tree, tabs=''):
    n_tabs = tabs + DEFAULT_TAB
    if type(tree) == str:
        print(tabs + ">", tree)
    elif tree is not None:
        for key in tree:
            print(tabs, key)
            printTree(tree[key], n_tabs)


cond_entropies = {}
intr_infos = {}
gain_ratios = {}
gains = {}

data, size = load_data("titanic-data.csv")
data.pop("Name")

res_class = "Survived"
id_attr = "PassengerId"
set_entropy = calc_set_entropy(data, res_class)


for attr in {k: data[k] for k in data if k not in (id_attr, res_class)}:
    cond_entropy, intr_info = calc_attrib(data, attr, res_class)
    gain = set_entropy - cond_entropy
    if intr_info != 0:
        gain_ratio = gain / intr_info
    else:
        gain_ratio = 10000.0

    gains[attr] = gain
    cond_entropies[attr], intr_infos[attr] = cond_entropy, intr_info
    gain_ratios[attr] = gain_ratio

    print("attr: %s, %f" % (attr, gain_ratio))


# order of the attributes (highest gain first)
attr_order = list({key: val for key, val in sorted(
    gains.items(), key=lambda ele: ele[1], reverse=True)}.keys())

tree = id3(data, res_class, attr_order)
printTree(tree)
# print(tree)
