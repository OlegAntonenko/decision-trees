import time
import random
import matplotlib.pyplot as plt
from c45 import C45
from genprogram import GP
from MannWhitneyTest import MannWhitneyU
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets


def despersion_and_average(results):
    m = sum(results) / len(results)
    varRes = sum([(xi - m) ** 2 for xi in results]) / len(results)
    return varRes, m


def split_data(data):
    trainingSample = []
    testSample = []
    random.shuffle(data)
    partLen = int(len(data) / 5)
    dataSplit = [data[partLen * k:partLen * (k + 1)] for k in range(5)]
    for i in range(len(dataSplit)):
        testSample.append(dataSplit[i])
        sumTraining = []
        for j in range(len(dataSplit)):
            if j != i:
                sumTraining += dataSplit[j]
        trainingSample.append(sumTraining)
    return trainingSample, testSample


def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()


tree = C45(maxDepth=4, split="best")
tree.extract_names(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat")
tree.extract_data(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat")
trainingSample, testSample = split_data(tree.get_data())

# Count average accuracy tree
# listAccuracy = []
# listAccuracySklearn = []
# num = 0
# for i, j in zip(trainingSample, testSample):
#     num += 1
#     tree.set_data(i)
#     start = time.clock()
#     tree.generate_tree()
#     end = time.clock()
#     print("Time generate tree " + str(num) + " : ", end - start)
#     tree.print_tree()
#     tree.preprocess_data(j)
#     listAccuracy.append(tree.accuracy(j))
#
#     # Use sklearn
#     estimator = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=4, splitter="random")
#     X_train = [x[:-1] for x in i]
#     Y_train = [y[-1] for y in i]
#     X_test = [x[:-1] for x in j]
#     Y_test = [y[-1] for y in j]
#     start = time.clock()
#     estimator.fit(X_train, Y_train)  # training decision tree
#     # plt.figure(figsize=(10,7))
#     # plot_tree(estimator)
#     # plt.show()
#     end = time.clock()
#     print("Time generate tree with sklearn " + str(num) + " : ", end - start, end="\n\n")
#     listAccuracySklearn.append(estimator.score(X_test, Y_test))  # accuracy tree
    
# averageAccuracy = sum(listAccuracy)/len(listAccuracy)
# averageAccuracySklearn = sum(listAccuracySklearn)/len(listAccuracySklearn)
# print("Average accuracy tree: ", averageAccuracy)
# print("Average accuracy tree with sklearn: ", round(averageAccuracySklearn, 2))

arr_ensamble = []
arr_best_split = []
arr_random_split = []
arr_sklearn_bagging = []
arr_sklearn_random_forest = []

for z in range(10):
    # Count average accuracy forest
    sizePopulation = 30

    listAccuracyGenTrees = []
    listAccuracyTrees_randomSplit = []

    listAccuracyGenTrees_BestSplit = []
    listAccuracyGenTrees_RandomSplit = []

    listAccuracyBagging_sklearn = []
    listAccuracyRandomForest_sklearn = []
    numKV = 0
    for i, j in zip(trainingSample, testSample):

        #####################################
        # list accuracy population with best split
        list_accuracy = []

        start = time.clock()

        genProgramm = GP(sizeForest=10, pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat", train_data=i,
                         test_data=j, split="best", maxDepth=5, alpha=0)
        genProgramm.generate_random_forest()

        listAccuracyGenTrees.append(genProgramm.accuracy_forest())
        print("Accuracy: ", listAccuracyGenTrees[len(listAccuracyGenTrees) - 1])

        num = 0
        while True:
            if num == sizePopulation:
                break
            genProgramm.tournament_selection_forest()
            genProgramm.crossing_forest()
            genProgramm.mutation_forest()
            genProgramm.fitness_function()
            list_accuracy.append(genProgramm.accuracy_forest())
            num += 1

        numKV += 1
        end = time.clock()
        print("Time generate forest with KV number " + str(numKV) + " : ", end - start)

        listAccuracyGenTrees_BestSplit.append(genProgramm.accuracy_forest())
        print("Accuracy with best split: ", listAccuracyGenTrees_BestSplit[len(listAccuracyGenTrees_BestSplit) - 1])

        # genProgramm.print_forest()

        list_accuracy = []

        start = time.clock()

        genProgramm = GP(sizeForest=10, pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat", train_data=i,
                         test_data=j, split="random", maxDepth=5, alpha=0)

        genProgramm.generate_random_forest()

        num = 0
        while True:
            if num == sizePopulation:
                break
            genProgramm.tournament_selection_forest()
            genProgramm.crossing_forest()
            genProgramm.mutation_forest()
            genProgramm.fitness_function()
            list_accuracy.append(genProgramm.accuracy_forest())
            num += 1

        end = time.clock()
        print("Time generate forest random KV number " + str(numKV) + " : ", end - start)

        listAccuracyGenTrees_RandomSplit.append(genProgramm.accuracy_forest())
        print("Accuracy with random split: ", listAccuracyGenTrees_RandomSplit[len(listAccuracyGenTrees_RandomSplit) - 1])

        # use sklearn
        X_train = [x[:-1] for x in i]
        Y_train = [y[-1] for y in i]
        X_test = [x[:-1] for x in j]
        Y_test = [y[-1] for y in j]

        clf = BaggingClassifier(n_estimators=10).fit(X_train, Y_train)
        listAccuracyBagging_sklearn.append(clf.score(X_test, Y_test))
        print("Accuracy bagging with use sklearn: ", clf.score(X_test, Y_test))

        clf2 = RandomForestClassifier(max_depth=7, n_estimators=10).fit(X_train, Y_train)
        listAccuracyRandomForest_sklearn.append(clf2.score(X_test, Y_test))
        print("Accuracy random forest with use sklearn: ", clf2.score(X_test, Y_test))

    # genProgramm.print_forest()
    averageAccuracyGenTrees = sum(listAccuracyGenTrees)/len(listAccuracyGenTrees)
    arr_ensamble.append(averageAccuracyGenTrees)
    print("Average accuracy forest", averageAccuracyGenTrees)

    averageAccuracyGenTrees_BestSplit = sum(listAccuracyGenTrees_BestSplit) / len(listAccuracyGenTrees_BestSplit)
    arr_best_split.append(averageAccuracyGenTrees_BestSplit)
    print("Average accuracy forest GP with best split: ", averageAccuracyGenTrees_BestSplit)

    averageAccuracyGenTrees_RandomSplit = sum(listAccuracyGenTrees_RandomSplit) / len(listAccuracyGenTrees_RandomSplit)
    arr_random_split.append(averageAccuracyGenTrees_RandomSplit)
    print("Average accuracy forest GP with random split: ", averageAccuracyGenTrees_RandomSplit)

    averageAccuracyBagging_sklearn = sum(listAccuracyBagging_sklearn) / len(listAccuracyBagging_sklearn)
    arr_sklearn_bagging.append(averageAccuracyBagging_sklearn)
    print("Average accuracy bagging sklearn: ", averageAccuracyBagging_sklearn)

    averageAccuracyRandomForest_sklearn = sum(listAccuracyRandomForest_sklearn) / len(listAccuracyRandomForest_sklearn)
    arr_sklearn_random_forest.append(averageAccuracyRandomForest_sklearn)
    print("Average accuracy random forest sklearn: ", averageAccuracyRandomForest_sklearn)


despersion_ensamble, average_ensamble = despersion_and_average(arr_ensamble)
print("despersion ensamble = ", despersion_ensamble)
print("average_ensamble = ", average_ensamble, end="\n")

despersion_best_split, average_best_split = despersion_and_average(arr_best_split)
print("despersion_best_split = ", despersion_best_split)
print("average_best_split = ", average_best_split, end="\n")

despersion_random_split, average_random_split = despersion_and_average(arr_random_split)
print("despersion_random_split = ", despersion_random_split)
print("average_random_split = ", average_random_split, end="\n")

despersion_sklearn_bagging, average_sklearn_bagging = despersion_and_average(arr_sklearn_bagging)
print("despersion_sklearn_bagging = ", despersion_sklearn_bagging)
print("average_sklearn_bagging = ", average_sklearn_bagging, end="\n")

despersion_sklearn_random_forest, average_sklearn_random_forest = despersion_and_average(arr_sklearn_random_forest)
print("despersion_sklearn_random_forest = ", despersion_sklearn_random_forest)
print("average_sklearn_random_forest = ", average_sklearn_random_forest, end="\n")

print("GP random split: ", arr_random_split)
print("GP best split: ", arr_best_split)
print("GP random split/GP best split: " + MannWhitneyU(arr_random_split, arr_best_split), end="\n")

print("GP random split: ", arr_random_split)
print("sklearn bagging: ", arr_sklearn_bagging)
print("GP random split/GP best split: " + MannWhitneyU(arr_random_split, arr_sklearn_bagging), end="\n")

print("GP random split: ", arr_random_split)
print("sklearn random forest: ", arr_sklearn_random_forest)
print("GP random split/GP best split: " + MannWhitneyU(arr_random_split, arr_sklearn_random_forest), end="\n")
