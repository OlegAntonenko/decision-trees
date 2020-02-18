from c45 import C45


tree = C45("C:\\Users\\Олег\\Documents\\Диплом\\data\\banana.dat")
tree.extract_data()
tree.preprocess_data()
tree.generate_tree()
tree.print_tree()

# iris = [2.9, 1.0, 3.4, 4.2]
# classObj = tree.use_tree(iris)
# print("Input: ", iris)
# print("Output: ", classObj)

banana = [1.14, -0.114]
classObj = tree.use_tree(banana)
print("Input: ", banana)
print("Output: ", classObj)
