import math
import copy
import random

import matplotlib.pyplot as plt

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots(2, 2, figsize=(12, 7))



    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax[0, 0].plot(x_data[0], y_data[0], lw=2, color='#539caf', alpha=1)
    ax[0, 1].plot(x_data[1], y_data[1], lw=2, color='red', alpha=1)
    ax[1, 0].plot(x_data[2], y_data[2], lw=2, color='orange', alpha=1)
    ax[1, 1].plot(x_data[3], y_data[3], lw=2, color='black', alpha=1)

    # Label the axes and provide a title
    # ax.set_title(title)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)

    plt.show()


class C45:

    def __init__(self, maxDepth=math.inf, split="best"):
        self.pathToData = []
        self.classes = []
        self.attrValues = {}
        self.numAttributes = -1
        self.attributes = []
        self.data = []
        self.tree = None
        self.gainArr = []
        self.maxDepth = maxDepth
        self.split = split
        self.depth = -1

    def set_tree(self, tree):
        self.tree = tree
        self.depth = tree.depth

    def get_tree(self):
        return self.tree

    def get_attributes(self):
        return self.attributes

    def get_classes(self):
        return self.classes

    def set_data(self, data):
        self.data = copy.deepcopy(data)
        self.preprocess_data(self.data)

    def get_data(self):
        return self.data

    def get_depth(self):
        return self.depth

    def extract_names(self, pathToData):
        self.pathToData = pathToData
        with open(self.pathToData, 'r') as file:
            data = file.read()
            data = data.split('\n')
            for i in range(1, len(data)):
                if data[i].split(' ')[0] != "@attribute":
                    break
                if data[i + 1].split(' ')[0] == "@attribute":
                    self.attributes.append(data[i].split(' ')[1])
                    if data[i][-1] == ']':
                        self.attrValues[self.attributes[len(self.attributes) - 1]] = ["continuous"]
                    else:
                        values = [i.strip('{}') for i in data[i].split(' ')[-1].split(',')]
                        self.attrValues[self.attributes[len(self.attributes) - 1]] = values
                else:
                    classes = data[i].split(' ')[2:]
                    for i in classes:
                        self.classes.append(i.strip('}{,'))
            self.numAttributes = len(self.attributes)

    def extract_data(self, pathToData):
        self.pathToData = pathToData
        with open(pathToData, 'r') as file:
            data = file.read()
            data = data.split('\n')
            for i in range(1, len(data)):
                if data[i].split(' ')[0] != "@attribute":
                    data = data[(i + 3):]
                    break
            for i in range(len(data)):
                row = [x.strip() for x in data[i].split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)  # add row in data
            self.preprocess_data(self.data)

    def preprocess_data(self, data):
        for index in range(len(data)):
            for attr_index in range(self.numAttributes):
                if (not self.is_attr_discrete(self.attributes[attr_index])):  # if view is "continuous" then
                    data[index][attr_index] = float(data[index][attr_index])  # convert string to float

    def is_attr_discrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def generate_tree(self):
        self.tree = self.recursive_generate_tree(self.data, self.attributes)
        self.tree.depth = self.depth

    def recursive_generate_tree(self, curdata, curattributes, depth=0):
        if len(curdata) == 0:
            return Node(True, "Fail", None)  # fail
        elif self.all_same_class(curdata) is not False:
            return Node(True, self.all_same_class(curdata), None)  # return a node with that class
        elif len(curattributes) == 0 or self.maxDepth == depth or len(curdata) <= 5:
            majclass = self.get_maj_class(curdata)  # return a node with the majority class
            return Node(True, majclass, None)
        elif self.uniformity(curdata, curattributes):
            return Node(True, "Fail", None)  # fail
        else:
            if self.split == "best":
                (best, best_threshold, splitted) = self.split_attribute(curdata, curattributes)
            elif self.split == "random":
                (best, best_threshold, splitted) = self.split_attribute_random(curdata, curattributes)
            # remainingAttributes = curattributes[:]
            # remainingAttributes.remove(best) # use attributes once
            if best == -1:
                return Node(True, "Fail", None)  # fail
            node = Node(False, best, best_threshold)
            depth += 1
            if depth > self.depth:
                self.depth = depth
            node.children = [self.recursive_generate_tree(subset, curattributes, depth) for subset in splitted]
            return node

    def split_attribute(self, curData, curAttributes):
        check = False

        splitted = []
        maxEnt = -math.inf
        best_attribute = -1
        best_threshold = None  # None for discrete attributes, threshold value for continuous attributes
        for attribute in curAttributes:
            indexOfAttribute = self.attributes.index(attribute)
            if self.is_attr_discrete(attribute):
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]
                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break
                e = self.gain(curData, subsets)
                if e > maxEnt:
                    maxEnt = e
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
            else:
                curData.sort(key=lambda x: x[indexOfAttribute])
                for j in range(0, len(curData)-1):
                    if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute] and curData[j][-1] != curData[j + 1][-1]:
                        check = True

                        threshold = (curData[j][indexOfAttribute] + curData[j + 1][indexOfAttribute]) / 2
                        less = []
                        greater = []
                        for row in curData:
                            if (row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])
                        if e >= maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_attribute = attribute
                            best_threshold = threshold

        if best_attribute == -1:
            if check:
                raise ValueError("best = -1")
            else:
                for attribute in curAttributes:
                    indexOfAttribute = self.attributes.index(attribute)
                    curData.sort(key=lambda x: x[indexOfAttribute])
                    for j in range(0, len(curData) - 1):
                        if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute]:
                            threshold = (curData[j][indexOfAttribute] + curData[j + 1][indexOfAttribute]) / 2
                            less = []
                            greater = []
                            for row in curData:
                                if row[indexOfAttribute] > threshold:
                                    greater.append(row)
                                else:
                                    less.append(row)
                            e = self.gain(curData, [less, greater])
                            if e >= maxEnt:
                                splitted = [less, greater]
                                maxEnt = e
                                best_attribute = attribute
                                best_threshold = threshold

        # if best_attribute == -1:
        #     raise ValueError("xxx")

        return best_attribute, best_threshold, splitted

    def split_attribute_random(self, curData, curAttributes):
        check = False

        arraySubsets = []
        arrayGain = [0]
        arrayThreshold = []
        splitted = []
        maxEnt = 0  # -math.inf
        best_attribute = -1
        best_threshold = None  # None for discrete attributes, threshold value for continuous attributes
        for attribute in curAttributes:
            indexOfAttribute = self.attributes.index(attribute)
            if self.is_attr_discrete(attribute):
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]
                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break
                e = self.gain(curData, subsets)
                arraySubsets.append(subsets)
                arrayGain.append(arrayGain[len(arrayGain) - 1] + e)
                arrayThreshold.append(None)
            else:
                curData.sort(key=lambda x: x[indexOfAttribute])
                for j in range(0, len(curData)-1):
                    if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute] and curData[j][-1] != curData[j + 1][-1]:
                        check = True

                        threshold = (curData[j][indexOfAttribute] + curData[j + 1][indexOfAttribute]) / 2
                        less = []
                        greater = []
                        for row in curData:
                            if (row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])
                        if e >= maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_threshold = threshold
                if splitted != []:
                    arraySubsets.append(splitted)
                    arrayGain.append(arrayGain[len(arrayGain) - 1] + maxEnt)
                    arrayThreshold.append(best_threshold)

        if len(arrayGain) == 0:
            if check:
                raise ValueError("best = -1")
            else:
                for attribute in curAttributes:
                    indexOfAttribute = self.attributes.index(attribute)
                    curData.sort(key=lambda x: x[indexOfAttribute])
                    for j in range(0, len(curData) - 1):
                        if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute]:
                            check = True

                            threshold = (curData[j][indexOfAttribute] + curData[j + 1][indexOfAttribute]) / 2
                            less = []
                            greater = []
                            for row in curData:
                                if (row[indexOfAttribute] > threshold):
                                    greater.append(row)
                                else:
                                    less.append(row)
                            e = self.gain(curData, [less, greater])
                            if e >= maxEnt:
                                splitted = [less, greater]
                                maxEnt = e
                                best_threshold = threshold
                    if splitted != []:
                        arraySubsets.append(splitted)
                        arrayGain.append(arrayGain[len(arrayGain) - 1] + maxEnt)
                        arrayThreshold.append(best_threshold)



        rnd = random.random() * arrayGain[-1]
        for gainInd in range(0, len(arrayGain)-1):  # choose attribute with rnd
            if arrayGain[gainInd] < rnd <= arrayGain[gainInd + 1]:
                best_attribute = curAttributes[gainInd]
                splitted = arraySubsets[gainInd]
                best_threshold = arrayThreshold[gainInd]
                break

        # if best_attribute == -1:
        #     raise ValueError("best = -1")

        return best_attribute, best_threshold, splitted

    def gain(self, unionSet, subsets):
        S = len(unionSet)
        # calculate impurity before split
        impurityBeforeSplit = self.entropy(unionSet)
        # calculate impurity after split
        weights = [len(subset) / S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i] * self.entropy(subsets[i])
        # calculate total gain
        totalGain = impurityBeforeSplit - impurityAfterSplit
        return totalGain

    def entropy(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in dataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        probabilities = [x / S for x in num_classes]
        ent = 0
        for num in probabilities:
            ent += num * self.log(num)
        return ent * -1

    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)

    def all_same_class(self, data):
        for row in data:
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def get_maj_class(self, curdata):
        freq = [0] * len(self.classes)
        for row in curdata:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]

    # def check(self, data, attributes):
    #     count = self.uniformity(data, attributes) + self.check_equal(data, attributes)
    #     if len(attributes) == count:
    #         return True
    #     return False
    #
    def uniformity(self, data, attributes):
        count = 0
        for attribute in attributes:
            indexOfAttribute = self.attributes.index(attribute)
            data.sort(key=lambda x: x[indexOfAttribute])
            if data[0][indexOfAttribute] == data[len(data) - 1][indexOfAttribute]:
                count += 1
        if len(attributes) == count:
            return True
        return False

    # def check_equal(self, data, attributes):
    #     count = 0
    #     for attribute in attributes:
    #         countEqual = [0 for i in self.attrValues[attribute]]
    #         indexofAttribute = self.attributes.index(attribute)
    #         for obj in data:
    #             countEqual[self.attrValues[attribute].index(obj[indexofAttribute])] += 1
    #         countEqual = [i for i in countEqual if i != 0]
    #         if len(set(countEqual)) == 1:
    #             count += 1
    #     if len(attributes) == count:
    #         return True
    #     return False

    def print_tree(self):
        self.print_node(self.tree)

    def print_node(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " + self.attrValues[str(node.label)][index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + self.attrValues[str(node.label)][index] + " : ")
                        self.print_node(child, indent + "	")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold) + " : ")
                    self.print_node(leftChild, indent + "	")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.print_node(rightChild, indent + "	")

    def use_tree(self, obj):
        classObj = self.check_node(obj, self.tree)
        return classObj

    def check_node(self, obj, node):
        if node.isLeaf:
            return node.label
        else:
            ind = self.attributes.index(node.label)
            if node.threshold is None:
                num = self.attrValues[node.label].index(obj[ind])
                classObj = self.check_node(obj, node.children[num])
            else:
                num = obj[ind]
                if num <= node.threshold:
                    classObj = self.check_node(obj, node.children[0])
                else:
                    classObj = self.check_node(obj, node.children[1])
            return classObj

    def accuracy(self, data):
        conformity = 0
        for i in data:
            classObj = self.use_tree(i[:-1])
            if classObj == i[-1]:
                conformity += 1
        return conformity/len(data)

    def mutation(self):
        self.mutation_check_node(self.tree, self.data, self.attributes)

    def mutation_check_node(self, node, curData, Attributes, depth=0):
        if not node.isLeaf:
            curAttributes = copy.copy(Attributes)
            if random.random() <= 0.5 and len(Attributes) > 1:
                curAttributes.remove(node.label)
                if len(curData) == 0:
                    node = Node(True, "Fail", None)  # fail
                elif self.all_same_class(curData) is not False:
                    node = Node(True, self.all_same_class(curData), None)  # return a node with that class
                elif self.maxDepth == depth or len(curData) <= 5:
                    majclass = self.get_maj_class(curData)  # return a node with the majority class
                    node = Node(True, majclass, None)
                elif self.uniformity(curData, curAttributes):
                    return Node(True, "Fail", None)  # fail
                else:
                    if self.split == "best":
                        (best, best_threshold, splitted) = self.split_attribute(curData, curAttributes)
                    elif self.split == "random":
                        (best, best_threshold, splitted) = self.split_attribute_random(curData, curAttributes)
                    remainingAttributes = curAttributes[:]
                    # remainingAttributes.remove(best)
                    remainingAttributes.append(node.label)
                    if best == -1:
                        return Node(True, "Fail", None)  # fail
                    depth = +1
                    node.label = best
                    node.threshold = best_threshold
                    node.children = [self.recursive_generate_tree(subset, remainingAttributes, depth)
                                     for subset in splitted]
            else:
                indexOfAttribute = self.attributes.index(node.label)
                if self.is_attr_discrete(node.label):
                    valuesForAttribute = self.attrValues[node.label]
                    subsets = [[] for a in valuesForAttribute]
                    for row in curData:
                        for index in range(len(valuesForAttribute)):
                            if row[indexOfAttribute] == valuesForAttribute[index]:
                                subsets[index].append(row)
                                break
                else:
                    less = []
                    greater = []
                    for row in curData:
                        if row[indexOfAttribute] >= node.threshold:
                            greater.append(row)
                        else:
                            less.append(row)
                    subsets = [less, greater]
                # curAttributes.remove(node.label)
                depth = +1
                [self.mutation_check_node(nodeChild, subset, curAttributes, depth)
                 for nodeChild, subset in zip(node.children, subsets) if len(subset) != 0]


class Node:

    def __init__(self, isLeaf, label, threshold, crossing=False, depth=False):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []
        self.crossing = crossing
        self.depth = depth
