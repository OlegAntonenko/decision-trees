from c45 import C45


class GP:

    def __init__(self, sizeForest, pathToData):
        self.forest = []
        self.pathToData = pathToData
        self.sizeForest = sizeForest
        self.data = []

    def generate_random_forest(self, data):
        self.forest = []
        self.data = data
        tree = C45(maxDepth=3, split="random")
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
        tree = C45(maxDepth=3, split="best")
        tree.extract_names(self.pathToData)
        tree.set_data(self.data)
        for t in self.forest:
            tree.set_tree(t)
            tree.mutation()
            forest.append(tree.get_tree())
        self.forest = forest[:]

    def crossing_forest(self):
        tree = C45()
