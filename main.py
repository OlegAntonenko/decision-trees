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


def cross_validation(data):
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


tree = C45(maxDepth=5)
tree.extract_names(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat")
tree.extract_data(pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat")

trainingSample, testSample = cross_validation(tree.get_data())

# Count average accuracy tree
listAccuracy = []
for i, j in zip(trainingSample, testSample):
    tree.set_data(i)
    start = time.clock()
    tree.generate_tree()
    tree.print_tree()
    end = time.clock()
    print("Time generate tree: ", end - start, end="\n\n")
    tree.preprocess_data(j)
    listAccuracy.append(tree.accuracy(j))
averageAccuracy = sum(listAccuracy)/len(listAccuracy)
print("Average accuracy tree: ", round(averageAccuracy, 2), end="\n\n")

# gainArray = tree.get_gain_array()
# numArray = []
# for i in range(len(gainArray)):
#     numArray.append([])
#     for j in range(len(gainArray[i])):
#         numArray[len(numArray) - 1].append(j)
# for i in range(len(numArray)):
#     lineplot(numArray[i], gainArray[i], "num", "gain", "Change gain")

genProgramm = GP(sizeForest=10, pathToData="C:\\Users\\Олег\\Documents\\Диплом\\data\\car.dat")

# Count average accuracy forest
listAccuracy = []
for i, j in zip(trainingSample, testSample):
    genProgramm.generate_random_forest(i)
    listAccuracy.append(genProgramm.accuracy_forest(j))
    print("Accuracy forest: " + str(round(listAccuracy[len(listAccuracy) - 1], 2)), end="\n\n")
averageAccuracy = sum(listAccuracy)/len(listAccuracy)
print("Average accuracy forest: ", averageAccuracy, end="\n\n")

# Use sklearn
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)  # 70% training and 30% test
clf = clf.fit(X_train, Y_train)
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
sample_one_pred = int(clf.predict([[5, 5, 1, 3]]))
print(sample_one_pred)
