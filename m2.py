import math
from collections import Counter

# Step 1: Calculate Entropy
def entropy(data):
    count = Counter(data)
    total = len(data)
    return sum([- (count[label] / total) * math.log2(count[label] / total) for label in count])

# Step 2: Calculate Information Gain
def information_gain(data, feature_index):
    # Get all unique values for this feature
    feature_values = set(row[feature_index] for row in data)
    
    # Calculate the weighted average entropy after splitting on this feature
    total_entropy = entropy([row[-1] for row in data])  # target class entropy
    weighted_entropy = 0
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy([row[-1] for row in subset])
    
    return total_entropy - weighted_entropy

# Step 3: Find the best feature to split on
def best_feature(data):
    # The number of features (excluding the target class)
    num_features = len(data[0]) - 1
    best_gain = -1
    best_feature_index = -1
    
    # For each feature, calculate information gain
    for i in range(num_features):
        gain = information_gain(data, i)
        if gain > best_gain:
            best_gain = gain
            best_feature_index = i
    
    return best_feature_index

# Step 4: Build the Decision Tree recursively
def id3(data, feature_names):
    # If all data points have the same label, return that label
    labels = [row[-1] for row in data]
    if len(set(labels)) == 1:
        return labels[0]
    
    # If no features are left, return the most common label
    if len(data[0]) == 1:
        return Counter(labels).most_common(1)[0][0]
    
    # Find the best feature to split on
    best_feature_index = best_feature(data)
    best_feature_name = feature_names[best_feature_index]
    
    # Create a new subtree with the best feature as the root
    tree = {best_feature_name: {}}
    
    # Split the data by the best feature's values and recurse
    feature_values = set(row[best_feature_index] for row in data)
    for value in feature_values:
        subset = [row for row in data if row[best_feature_index] == value]
        subtree = id3([row[:best_feature_index] + row[best_feature_index+1:] for row in subset], feature_names[:best_feature_index] + feature_names[best_feature_index+1:])
        tree[best_feature_name][value] = subtree
    
    return tree

# Dataset (features + target class 'Play Tennis')
data = [
    ['Sunny', 'Hot', 'No', 'No'],
    ['Sunny', 'Hot', 'Yes', 'No'],
    ['Overcast', 'Mild', 'No', 'Yes'],
    ['Rainy', 'Cool', 'No', 'Yes'],
    ['Rainy', 'Cool', 'Yes', 'No'],
    ['Overcast', 'Hot', 'Yes', 'Yes'],
    ['Sunny', 'Mild', 'No', 'Yes'],
    ['Sunny', 'Cool', 'No', 'Yes'],
    ['Rainy', 'Mild', 'No', 'Yes'],
    ['Sunny', 'Mild', 'Yes', 'No'],
    ['Overcast', 'Mild', 'Yes', 'Yes'],
    ['Rainy', 'Mild', 'Yes', 'No']
]

# Feature names
feature_names = ['Weather', 'Temperature', 'Windy']

# Build the decision tree
tree = id3(data, feature_names)

# Output the decision tree
print("Decision Tree:", tree)
