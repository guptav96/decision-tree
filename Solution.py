import numpy as np
import pandas as pd
# import pprint
import copy

# list of attribute names
columns = ['A' + str(i) for i in range(1,17)]
# mentioned in description.txt
continuous_attributes = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
# default label
default_class = '+'
# post-pruning tree defaults to false.
postprune = False
order_attributes_tree = []

# store the current best tree in a global variable
current_best_tree = {}

train_df = pd.read_csv('train.txt', header=None, sep="\t")
validation_df = pd.read_csv('validation.txt', header=None, sep="\t")
test_df = pd.read_csv('test.txt', header=None, sep="\t")
train_df.columns = test_df.columns = validation_df.columns = columns

def plot():
    preprocessed_test_df = preprocess(test_df)
    preprocessed_validation_df = preprocess(validation_df)
    original_validation_labels = validation_df.iloc[:,-1]
    original_testing_labels = test_df.iloc[:,-1]
    validation_accuracy = []
    test_accuracy = []
    depth_range = np.arange(2,16,1)
    for depth in depth_range:
        DecisionTreeBounded(depth)
        validation_accuracy.append(accuracy_of_tree(current_best_tree, preprocessed_validation_df, original_validation_labels))
        test_accuracy.append(accuracy_of_tree(current_best_tree, preprocessed_test_df, original_testing_labels))
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2, figsize=(10,5))
    # fig.suptitle('Validation and Testing Data Accuracy')
    # axs[0].plot(validation_accuracy)
    # axs[1].plot(test_accuracy)
    # axs[0].set_title('Validation Accuracy vs Max Depth')
    # axs[1].set_title('Test Accuracy vs Max Depth')
    # plt.xticks(range(2,16,1))
    # for ax in axs.flat:
    #     ax.set(xlabel='max-depth', ylabel='accuracy')
    plt.title('Validation and Testing Accuracy vs Max Depth (Decision Trees)')
    plt.plot(depth_range, validation_accuracy)
    plt.plot(depth_range, test_accuracy)
    plt.legend(['Validation Data', 'Test Data'])
    plt.show()

def preprocess(df):
    # replace missing attributes with mode if categorical attributes,
    # or with mean in case of continuous attributes
    df = df.replace('?', np.nan)
    for column in df.columns[:-1]:
        # print(column)
        if column in continuous_attributes:
            df[column] = df[column].astype(float)
            df[column].fillna(value = df[column].mean(), inplace=True)
        else:
            df[column].fillna(value = df[column].mode()[0], inplace=True)
    return df

def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    # print(elements, counts)
    entropy = np.sum([(-count/np.sum(counts))*np.log2(count/np.sum(counts)) for count in counts])
    return entropy

def information_gain_continuous(data, split_attribute, label):
    total_entropy = entropy(data[label])
    lowest_entropy = 1 # initialize to 1
    split_value = 0 # initialize to 0

    sorted_data = data.sort_values(by=split_attribute)
    num_elem = data[split_attribute].shape[0]
    for idx in range(1, num_elem, 1):
        # check the information gain for all possible splits and choose the max IG
        if not sorted_data.iloc[idx][label] == sorted_data.iloc[idx-1][label]:
            temp_split_val = sorted_data.iloc[idx][split_attribute]
            weighted_entropy = 1/num_elem* \
                                (idx * entropy(sorted_data[sorted_data[split_attribute] <= temp_split_val][label]) + \
                                (num_elem - idx) * entropy(sorted_data[sorted_data[split_attribute] > temp_split_val][label]))
            if weighted_entropy < lowest_entropy:
                lowest_entropy = weighted_entropy
                split_value = temp_split_val
    return total_entropy - lowest_entropy, split_value

def information_gain(data, split_attribute, label):
    # print(split_attribute, len(data))
    if split_attribute in continuous_attributes:
        return information_gain_continuous(data, split_attribute, label)

    total_entropy = entropy(data[label])

    elements, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum(counts[i]/np.sum(counts)*entropy(data[data[split_attribute] == elements[i]][label]) for i in range(len(elements)))

    return total_entropy - weighted_entropy, 0

def predict_query(query, tree):
    try:
        for tree_key, tree_value in tree.items():
            attrs = list(tree_value.keys())
            if tree_key in continuous_attributes:
                split_value = 0
                for attr in attrs:
                    if attr[0] == '>':
                        split_value = float(attr[1:])
                if float(query[tree_key]) <= split_value:
                    result = tree[tree_key]['<=' + str(split_value)]
                else:
                    result = tree[tree_key]['>' + str(split_value)]
            else:
                if query[tree_key] in attrs:
                    result = tree[tree_key][query[tree_key]]
                else:
                    result = default_class
            # recursively call predict on subtrees until you reach leaf node
            if isinstance(result, dict):
                result = predict_query(query, result)
    except:
        result = default_class
    return result

def predict(input_df, tree):
    predicted_list = []
    for i in range(len(input_df)):
        predicted_result = predict_query(input_df.iloc[i,:], tree)
        predicted_list.append(predicted_result)
    return predicted_list

def accuracy_of_tree(tree, input, output):
    total_num = len(input)
    pred_output = predict(input, tree)
    match = 0
    for i in range(total_num):
        if(pred_output[i] == output[i]):
            match += 1
    return float(match)/total_num

def preorder(temp_tree, attribute):
    if isinstance(temp_tree, dict):
        curr_attr = list(temp_tree.keys())[0]
        if curr_attr == attribute:
            values = list(temp_tree[attribute].keys())
            for value in values:
                temp_tree[attribute][value] = temp_tree[attribute]['best_class']
        else:
            tree_values = temp_tree.values()
            for value in tree_values:
                for value_values in list(value.values()):
                    if isinstance(value_values, dict):
                        preorder(value_values, attribute)
    return temp_tree

def post_prune(validation_df, tree):
    order_attributes_tree.reverse()
    for attr in order_attributes_tree:
        temp_tree = copy.deepcopy(tree)
        preorder(temp_tree, attr)
        accuracy_before_pruning = accuracy_of_tree(tree, validation_df.iloc[:,:-1], validation_df.iloc[:,-1])
        accuracy_after_pruning  = accuracy_of_tree(temp_tree, validation_df.iloc[:,:-1], validation_df.iloc[:,-1])
        current_gain = accuracy_after_pruning - accuracy_before_pruning
        if current_gain > 0:
            tree = temp_tree
    return tree

def ID3(data, attributes, label, maxDepth = None):
    if maxDepth == None:
        maxDepth = data.shape[0]
    global order_attributes_tree
    # print(data.shape[0])
    # keep track of most frequency label in the current dataset
    count_pos = np.count_nonzero(data[label] == '+')
    count_neg = len(data[label]) - count_pos
    most_freq_label = '+' if count_pos >= count_neg else '-'

    if len(attributes) == 0 or len(data) == 0:
        return most_freq_label
    # If all the examples have the same label, return a single node tree with the label
    elif len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
    else:
        # Select the attribute with the most information gain
        attribute_points = [information_gain(data, attribute, label)[0] for attribute in attributes]
        # print(attribute_points)
        best_gain = np.max(attribute_points)
        best_attribute_index = np.argmax(attribute_points)
        best_attribute = attributes[best_attribute_index]
        # create a root node for the tree with the best attribute
        tree = {best_attribute:{}}
        if best_gain < 0.05: # pre-pruning to some extent
            return {}
        # delete the best attribute from attributes array
        attributes = np.delete(attributes, best_attribute_index, 0)

        tree[best_attribute]['best_class'] = most_freq_label
        order_attributes_tree.append(best_attribute)

        if best_attribute in continuous_attributes:
            # for continuous attribute, split into two subtrees based on split value
            split_value = information_gain(data, best_attribute, label)[1]
            # print(type(split_value), split_value)
            data_left = data[data[best_attribute] <= split_value]
            data_right = data[data[best_attribute] > split_value]
            if len(data_left) == 0 or maxDepth <= 1:
                tree[best_attribute]['<=' + str(split_value)] = most_freq_label
            else:
                sub_tree = ID3(data_left, attributes, label, maxDepth - 1)
                tree[best_attribute]['<=' + str(split_value)] = sub_tree
            if len(data_right) == 0 or maxDepth <= 1:
                tree[best_attribute]['>' + str(split_value)] = most_freq_label
            else:
                sub_tree = ID3(data_right, attributes, label, maxDepth - 1)
                tree[best_attribute]['>' + str(split_value)] = sub_tree
        else: # for categorical and boolean attributes
            # For each possible value of attribute, grow a branch
            for val in np.unique(data[best_attribute]):
                data_v = data[data[best_attribute] == val]
                if(len(data_v) == 0) or maxDepth <= 1:
                    tree[best_attribute][val] = most_freq_label
                else:
                    # add the subtree
                    sub_tree = ID3(data_v, attributes, label, maxDepth - 1)
                    tree[best_attribute][val] = sub_tree
    # return the root node of the tree
    return tree

def DecisionTree():
    # call decision tree bounded function with max depth as none
    return DecisionTreeBounded(None)

def DecisionTreeBounded(maxDepth):
    # apply preprocessing separately to avoid any data leakage
    global current_best_tree
    preprocessed_train_df = preprocess(train_df)
    preprocessed_test_df = preprocess(test_df)
    preprocessed_validation_df = preprocess(validation_df)

    tree = ID3(preprocessed_train_df, columns[:-1], 'A16', maxDepth)

    if postprune:
        tree = post_prune(preprocessed_validation_df, tree)
    # pprint.pprint(tree)

    current_best_tree = tree
    original_training_labels = train_df.iloc[:,-1].tolist()
    predicted_training_labels  = predict(preprocessed_train_df.iloc[:,:-1], tree)
    original_testing_labels = test_df.iloc[:,-1].tolist()
    predicted_testing_labels  = predict(preprocessed_test_df.iloc[:,:-1], tree)

    labels = [original_training_labels, predicted_training_labels, original_testing_labels, predicted_testing_labels]
    # print(labels)
    return labels
