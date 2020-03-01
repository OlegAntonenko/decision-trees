import time
import matplotlib.pyplot as plt
from c45 import C45
from UseData import UseData


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


tree = C45("C:\\Users\\Олег\\Documents\\Диплом\\data\\banana.dat")
dataWorker = UseData()
dataWorker.extract_data("C:\\Users\\Олег\\Documents\\Диплом\\data\\banana.dat")
dataYX = dataWorker.split_data()
tree.extract_names()
listAccuracy = []
for YX in dataYX:
    tree.set_data(YX[1])
    # start = time.clock()
    tree.generate_tree()
    tree.preprocess_data(YX[0]) #test
    listAccuracy.append(tree.accuracy(YX[0]))
    # end = time.clock()
averageAccuracy = sum(listAccuracy)/len(listAccuracy)
print(averageAccuracy)

# print("Time generate tree: ", end - start)
# tree.print_tree()

# gainArray = tree.get_gain_array()
# numArray = []
# for i in range(len(gainArray)):
#     numArray.append([])
#     for j in range(len(gainArray[i])):
#         numArray[len(numArray) - 1].append(j)

# for i in range(len(numArray)):
#     lineplot(numArray[i], gainArray[i], "num", "gain", "Change gain")

# iris = [[2.9, 1.0, 3.4, 4.2, "hello, world"]]
# accuracy = tree.accuracy(iris)
# print("accuracy: ", accuracy)

# banana = [1.14, -0.114]
# classObj = tree.use_tree(banana)
# print("Input: ", banana)
# print("Output: ", classObj)
