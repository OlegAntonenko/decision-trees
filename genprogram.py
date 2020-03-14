from c45 import C45
from usedata import UseData


class GP:

    def __init__(self, sizeForest):
        self.forest = []
        self.sizeForest = sizeForest

    def generate_random_forest(self, pathToData):
        tree = C45()
        tree.extract_names(pathToData)
        dataWorker = UseData()
        dataWorker.extract_data(pathToData)
        data = dataWorker.get_data()
        tree.set_data(data)
        for i in range(self.sizeForest):
            tree.generate_tree()
            self.forest.append(tree.get_tree())
        print(self.forest)

    # def accuracy_forest(self):
