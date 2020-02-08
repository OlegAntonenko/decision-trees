from c45 import C45


tree = C45("iris-data.txt", "iris-names.txt")
tree.extract_data()
tree.preprocess_data()
tree.generate_tree()
tree.print_tree()
