import time
import random
import matplotlib.pyplot as plt
from c45 import C45
from genprogram import GP
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
# from usedata import UseData


def split_data(data):
    trainingSample = []
    testSample = []
    random.shuffle(data)
    partLen = int(len(data) / 10)
    dataSplit = [data[partLen * k:partLen * (k + 1)] for k in range(10)]
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


tree = C45(maxDepth=4)
tree.extract_names(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")
tree.extract_data(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")

trainingSample, testSample = split_data(tree.get_data())

# Count average accuracy tree
listAccuracy = []
listAccuracySklearn = []
num = 0
for i, j in zip(trainingSample, testSample):
    num += 1
    tree.set_data(i)
    start = time.clock()
    tree.generate_tree()
    end = time.clock()
    print("Time generate tree " + str(num) + " : ", end - start)
    tree.preprocess_data(j)
    listAccuracy.append(tree.accuracy(j))

    # Use sklearn
    estimator = DecisionTreeClassifier(random_state=0)
    X_train = [x[:-1] for x in i]
    Y_train = [y[-1] for y in i]
    X_test = [x[:-1] for x in j]
    Y_test = [y[-1] for y in j]
    start = time.clock()
    estimator.fit(X_train, Y_train)  # training decision tree
    end = time.clock()
    print("Time generate tree with sklearn " + str(num) + " : ", end - start, end="\n\n")
    listAccuracySklearn.append(estimator.score(X_test, Y_test))  # accuracy tree

averageAccuracy = sum(listAccuracy)/len(listAccuracy)
averageAccuracySklearn = sum(listAccuracySklearn)/len(listAccuracySklearn)
print("Average accuracy tree: ", round(averageAccuracy, 2))
print("Average accuracy tree with sklearn: ", round(averageAccuracySklearn, 2), end="\n\n")

# gainArray = tree.get_gain_array()
# numArray = []
# for i in range(len(gainArray)):
#     numArray.append([])
#     for j in range(len(gainArray[i])):
#         numArray[len(numArray) - 1].append(j)
# for i in range(len(numArray)):
#     lineplot(numArray[i], gainArray[i], "num", "gain", "Change gain")

genProgramm = GP(sizeForest=10, pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")

# Count average accuracy forest
listAccuracy = []
for i, j in zip(trainingSample, testSample):
    genProgramm.generate_random_forest(i)
    listAccuracy.append(genProgramm.accuracy_forest(j))
    print("Accuracy forest: " + str(round(listAccuracy[len(listAccuracy) - 1], 2)), end="\n\n")
averageAccuracy = sum(listAccuracy)/len(listAccuracy)
print("Average accuracy forest: ", averageAccuracy)
