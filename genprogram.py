from c45 import C45, Node
import math
import random


class GP:

    def __init__(self, sizeForest, pathToData, train_data, test_data, split="best", maxDepth=math.inf,
                 choice_quality="accuracy"):
        self.forest = []
        self.forest_child = []
        self.pathToData = pathToData
        self.sizeForest = sizeForest
        self.split = split
        self.maxDepth = maxDepth
        self.train_data = train_data
        self.test_data = test_data
        self.choice_quality = choice_quality

    def bootstrap(self):
        bootstrap_data = []
        for i in range(len(self.train_data)):
            bootstrap_data.append(random.choice(self.train_data))
        return bootstrap_data

    def generate_random_forest(self):
        self.forest = []
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        for i in range(self.sizeForest):
            bootstrap_data = self.bootstrap()
            tree.set_data(bootstrap_data)
            tree.generate_tree()
            self.forest.append(tree.get_tree())

    def fitness_function(self):
        self.forest += self.forest_child
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        accuracy_list = []
        for obj in self.forest:
            tree.set_tree(obj)
            if tree.depth == False:
                raise ValueError("depth = False")
            if "accuracy" == self.choice_quality:
                accuracy_list.append(tree.accuracy(self.test_data))
            elif "depth" == self.choice_quality:
                accuracy_list.append(self.quality_functional(tree.accuracy(self.train_data), tree.get_depth()))
            else:
                raise NameError("self.choice_quality")
        quality_list = [[i, j] for i, j in zip(accuracy_list, self.forest)]
        quality_list.sort(key=lambda x: x[0])
        self.forest = [i[1] for i in quality_list]
        self.forest = self.forest[len(self.forest_child)::]

    def quality_functional(self, accuracy, depth):
        fitness = 100*accuracy- depth
        return fitness

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

    def accuracy_forest(self):
        conformity = 0
        for i in self.test_data:
            classObj = self.use_forest(i[:-1])
            if classObj == i[-1]:
                conformity += 1
        return conformity / len(self.test_data)

    def mutation_forest(self):
        forest = []
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        tree.set_data(self.train_data)
        for t in self.forest_child:
            if tree.depth == False:
                raise ValueError("depth = False")
            tree.set_tree(t)
            tree.mutation()
            forest.append(tree.get_tree())
        self.forest_child = forest[:]

    def crossing_forest(self):
        forest = []
        while True:
            listIndex = [i for i in range(len(self.forest_child))]
            treeDad = self.forest_child.pop(random.choice(listIndex))

            listIndex = [i for i in range(len(self.forest_child))]
            treeMom = self.forest_child.pop(random.choice(listIndex))

            # crossing
            self.crossing(treeDad, treeMom)

            forest.append(treeDad)
            forest.append(treeMom)

            if len(self.forest_child) <= 1:
                break
        self.forest_child = forest[:]

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

    def tournament_selection_forest(self):
        self.forest_child = []
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        tree.extract_names(self.pathToData)
        while True:
            best_accuracy = -1
            for i in range(2):
                listIndex = [i for i in range(len(self.forest))]
                structure_tree = self.forest[random.choice(listIndex)]
                tree.set_tree(structure_tree)
                accuracy = tree.accuracy(self.test_data)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = structure_tree

            self.forest_child.append(self.copy_tree(best_tree))

            if len(self.forest) == len(self.forest_child):
                break
        # self.forest = forest[:]

    def copy_tree(self, tree):
        copy_tree = self.copy_node(tree)
        return copy_tree

    def copy_node(self, node):
        if node.isLeaf is False:
            node_copy = Node(node.isLeaf, node.label, node.threshold, depth=node.depth)
            node_copy.children = [self.copy_node(i) for i in node.children]
            return node_copy
        else:
            return Node(node.isLeaf, node.label, node.threshold)

    def print_forest(self):
        tree = C45(maxDepth=self.maxDepth, split=self.split)
        for i in range(len(self.forest)):
            tree.set_tree(self.forest[i])
            print()
            tree.print_tree()
