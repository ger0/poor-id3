#!/bin/python

import csv
import math

FILENAME = "titanic-homework.csv"
DEFAULT_TAB = "   "


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


def check_known_attribs(i, curr_attr, data, known_attr: {}):
    for chk in known_attr:
        if curr_attr != chk and data[chk][i] not in known_attr[chk]:
            return False
    return True


def check_inner_dict_empty(dict):
    for x in dict:
        if not bool(dict[x]):
            return True
    return False


def id(data, res, order=[], known_attr={}):
    # if there are no remaining attributes to check return nothing
    if not bool(order):
        return {}

    dict = {}
    attr = order[0]

    last = None
    are_identical = False

    # iterate over every entry in the data
    for i in range(len(data[attr])):
        new_known_attr = known_attr.copy()
        new_known_attr.setdefault(attr, [])

        # val is value of the i-th element in the attribute attr
        val = data[attr][i]
        result_val = data[res][i]

        # known_attr contains values known in the higher nodes
        # ignore entry if the attributes were matched in higher nodes
        if not check_known_attribs(i, attr, data, new_known_attr):
            continue

        if val not in new_known_attr[attr]:
            new_known_attr[attr].append(val)

        next = id(data, res, order[1:], new_known_attr)

        if last is None:
            last = result_val
            are_identical = True

        elif result_val != last:
            are_identical = False

        # check if the dict is empty
        if bool(next):
            # check if theres an empty dict inside
            if (check_inner_dict_empty(next)):
                dict[val] = {res: result_val}
            else:
                dict[val] = next
        else:
            dict[val] = {res: result_val}

    if (are_identical):
        return {res: last}
    else:
        return {attr: dict}


def printTree(tree, tabs=''):
    n_tabs = tabs + DEFAULT_TAB
    if type(tree) == str:
        print(tabs + tree)
    else:
        for key in tree:
            print(n_tabs + key)
            if bool(tree):
                printTree(tree[key], n_tabs)
            else:
                print(n_tabs + tree[key])


cond_entropies = {}
intr_infos = {}
gain_ratios = {}
gains = {}

# data, size = loadData(FILENAME)
# data.pop("Name")
# print(calcEntropy(data, attr="Pclass", res="Survived"))

data, size = load_data("./test.csv")
set_entropy = calc_set_entropy(data, res="decision")


for attr in {k: data[k] for k in data if k not in ('id', 'decision')}:
    cond_entropy, intr_info = calc_attrib(data, attr, res="decision")
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

tree = id(data, "decision", attr_order)
printTree(tree)
# print(tree)
