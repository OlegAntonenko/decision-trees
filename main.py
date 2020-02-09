from c45 import C45


tree = C45("C:\\Users\\Олег\\Documents\\Диплом\\data\\iris-data.txt",
           "C:\\Users\\Олег\\Documents\\Диплом\\data\\iris-names.txt")
tree.extract_data()
tree.preprocess_data()
tree.generate_tree()
tree.print_tree()

# Example
iris = [2.9, 1.0, 3.4, 4.2]
classObj = tree.use_tree(iris)
print(classObj)
