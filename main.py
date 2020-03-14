import time
import matplotlib.pyplot as plt
from c45 import C45
from usedata import UseData
from genprogram import GP


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


# tree = C45()
# tree.extract_names("C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")
# dataWorker = UseData()
# dataWorker.extract_data("C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")
# trainingSample, testSample = dataWorker.cross_validation()

# Count average accuracy
# listAccuracy = []
# for i, j in zip(trainingSample, testSample):
#     tree.set_data(i)
#     start = time.clock()
#     tree.generate_tree()
#     tree.print_tree()
#     end = time.clock()
#     print("Time generate tree: ", end - start, end="\n\n")
#     tree.preprocess_data(j)  # test
#     listAccuracy.append(tree.accuracy(j))
#
# averageAccuracy = sum(listAccuracy)/len(listAccuracy)
# print("Average accuracy: ", averageAccuracy)

# gainArray = tree.get_gain_array()
# numArray = []
# for i in range(len(gainArray)):
#     numArray.append([])
#     for j in range(len(gainArray[i])):
#         numArray[len(numArray) - 1].append(j)
# for i in range(len(numArray)):
#     lineplot(numArray[i], gainArray[i], "num", "gain", "Change gain")

genProgramm = GP(10)
genProgramm.generate_random_forest("C:\\Users\\Олег\\Documents\\Диплом\\data\\iris.dat")
