from c45 import C45
from usedata import UseData
from collections import Counter


class GP:

    def __init__(self, sizeForest):
        self.forest = []
        self.pathToData = []
        self.sizeForest = sizeForest

    def generate_random_forest(self, pathToData):
        self.pathToData = pathToData
        tree = C45(3)
        tree.extract_names(self.pathToData)
        dataWorker = UseData()
        dataWorker.extract_data(self.pathToData)
        data = dataWorker.get_data()
        tree.set_data(data)
        for i in range(self.sizeForest):
            tree.generate_tree()
            self.forest.append(tree.get_tree())
        print(self.forest)

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
