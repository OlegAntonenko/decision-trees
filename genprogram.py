from c45 import C45
import math
import random


class GP:

    def __init__(self, sizeForest, pathToData, split="best", maxDepth=math.inf):
        self.forest = []
        self.pathToData = pathToData
        self.sizeForest = sizeForest
        self.data = []
        self.split = split
        self.maxDepth = maxDepth

    # def bootstrap(self, data):
    #

    def generate_random_forest(self, data):
        self.forest = []
        self.data = data
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        tree.set_data(self.data)
        for i in range(self.sizeForest):
            tree.generate_tree()
            self.forest.append(tree.get_tree())

    def use_forest(self, obj):
        arrayAnswer = []
        tree = C45()
        tree.extract_names(self.pathToData)
        for t in self.forest:
            tree.set_tree(t)
            arrayAnswer.append(tree.use_tree(obj))
        classes = tree.get_classes()
        countObj = [0 for i in classes]
        for i in arrayAnswer:
            if i != "Fail":
                countObj[classes.index(i)] += 1
        classObj = classes[countObj.index(max(countObj))]
        return classObj

    def accuracy_forest(self, data):
        conformity = 0
        for i in data:
            classObj = self.use_forest(i[:-1])
            if classObj == i[-1]:
                conformity += 1
        return conformity / len(data)

    def mutation_forest(self):
        forest = []
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        tree.set_data(self.data)
        for t in self.forest:
            tree.set_tree(t)
            tree.mutation()
            forest.append(tree.get_tree())
        self.forest = forest[:]

    def crossing_forest(self):
        forest = []
        while True:
            listIndex = [i for i in range(len(self.forest))]
            treeDad = self.forest.pop(random.choice(listIndex))

            listIndex = [i for i in range(len(self.forest))]
            treeMom = self.forest.pop(random.choice(listIndex))

            # crossing
            self.crossing(treeDad, treeMom)

            forest.append(treeDad)
            forest.append(treeMom)

            if len(self.forest) == 0:
                break
        self.forest = forest[:]

    def crossing(self, treeDad, treeMom):
        addressesMom = self.get_addresses(treeMom)
        addressesDad = self.get_addresses(treeDad)

        addressCrossingMomToDad = random.choice(addressesMom)
        addressCrossingMomToDad.crossing = True

        addressCrossingDadToMom = random.choice(addressesDad)
        addressCrossingDadToMom.crossing = True

        self.change_address(treeMom, addressCrossingDadToMom)
        self.change_address(treeDad, addressCrossingMomToDad)

    def get_addresses(self, tree):
        addresses = []
        [self.get_address(child, addresses) for child in tree.children]
        return addresses

    def get_address(self, node, addresses):
        if not node.isLeaf:
            node.crossing = False
            addresses.append(node)
            [self.get_address(child, addresses) for child in node.children]

    def change_address(self, node, address):
        if not node.isLeaf:
            for i in range(len(node.children)):
                if node.children[i].crossing:
                    node.children[i] = address
                else:
                    self.change_address(node.children[i], address)

    def tournament_selection_forest(self, dataTest):
        forest = []
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        while True:
            best_accuracy = -1
            for i in range(2):
                listIndex = [i for i in range(len(self.forest))]
                structure_tree = self.forest.pop(random.choice(listIndex))
                tree.set_tree(structure_tree)
                accuracy = tree.accuracy(dataTest)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = structure_tree

            forest.append(best_tree)

            if len(self.forest) == 0:
                break
        self.forest = forest[:]

    def print_forest(self):
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        for i in range(len(self.forest)):
            tree.set_tree(self.forest[i])
            print()
            tree.print_tree()
