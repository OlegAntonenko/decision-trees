import math
import sys


class C45:

    def __init__(self, pathToData, pathToNames):
        self.pathToData = pathToData
        self.pathToNames = pathToNames
        self.classes = []
        self.attrValues = {}
        self.numAttributes = -1
        self.attributes = []
        self.data = []
        self.tree = None

    def extract_data(self):
        with open(self.pathToNames, 'r') as file:
            classes = file.readline()
            self.classes = [x.strip() for x in classes.split(",")]  # list name of classes
            for line in file:
                try:
                    [attribute, values] = [x.strip() for x in line.split(":")]  # lists of signs and view
                except ValueError:
                    print("Error: lists and his view split \":\" ")
                    sys.exit()
                values = [x.strip() for x in values.split(",")]
                self.attrValues[attribute] = values  # dictionary {'signs': view}
        self.numAttributes = len(self.attrValues.keys())  # number keys
        self.attributes = list(self.attrValues.keys())  # list of keys
        with open(self.pathToData, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)  # add row in data

    def preprocess_data(self):
        for index in range(len(self.data)):
            for attr_index in range(self.numAttributes):
                if (not self.is_attr_discrete(self.attributes[attr_index])):  # if view is "continuous" then
                    self.data[index][attr_index] = float(self.data[index][attr_index])  # convert string to float

    def is_attr_discrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def generate_tree(self):
        self.tree = self.recursive_generate_tree(self.data, self.attributes)

    def recursive_generate_tree(self, curdata, curattributes):
        allsame = self.all_same_class(curdata)  # return name class if all data have same class
        if len(curdata) == 0:
            return Node(True, "Fail", None)  # fail
        elif allsame is not False:
            return Node(True, allsame, None)  # return a node with that class
        elif len(curattributes) == 0:
            majclass = self.get_maj_class(curdata)  # return a node with the majority class
            return Node(True, majclass, None)
        else:
            (best, best_threshold, splitted) = self.split_attribute(curdata, curattributes)
            remainingAttributes = curattributes[:]
            remainingAttributes.remove(best)
            node = Node(False, best, best_threshold)
            node.children = [self.recursive_generate_tree(subset, remainingAttributes) for subset in splitted]
            return node

    def split_attribute(self, curData, curAttributes):
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
                    if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute]:
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
        return (best_attribute, best_threshold, splitted)

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

    def use_tree(self):
        pass


class Node:

    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []