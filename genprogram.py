from c45 import C45
from usedata import UseData


class GP:

    def __init__(self, sizeForest, pathToData):
        self.forest = []
        self.pathToData = pathToData
        self.sizeForest = sizeForest

    def generate_random_forest(self, data):
        self.forest = []
        tree = C45(maxDepth=4, split="random")
        tree.extract_names(self.pathToData)
        tree.set_data(data)
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
        tree = C45()
        tree.extract_names(self.pathToData)
        for t in self.forest:
            tree.set_tree(t)
            print("before")
            tree.print_tree()
            tree.mutation()
            print("after")
            tree.print_tree()
            print()
