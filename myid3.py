# Module file for implementation of ID3 algorithm.

import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import pickle

class DecisionTree:
    
    def __init__(self, load_from=None):
        self.tree = {}
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.tree = pickle.load(load_from)

    def Entropy(self, y):
        unique_elements, unique_count = np.unique(y, return_counts = True)
        entropy=np.sum([(-unique_count[i]/np.sum(unique_count))*np.log2(unique_count[i]/np.sum(unique_count))for i in range(len(unique_elements))])
        return entropy
    
    def Information_Gain(self, X, y, attribute):
        Total_entropy = self.Entropy(y)
        unique_elements, unique_count = np.unique(X, return_counts = True)
        Weight_entropy = np.sum([(unique_count[i]/np.sum(unique_count))*self.Entropy(X.where(X[attribute]==unique_elements[i])) for i in range(len(unique_elements))])
        Information_gain = Total_entropy - Weight_entropy
        return Information_gain
        
    def train(self, X, y, attrs, prune=False):
        tree = {}
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
    
        if len(np.unique(y)) <= 1:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
        else:
            pass
        
        if len(X) == 0:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]  
    
        if len(attrs) == 0:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]  
        else:
            ig_values_attrs = [self.Information_Gain(X, y, attribute) for attribute in attrs]
            index = np.argmax(ig_values_attrs)
            best_attribute = attrs[index]
            tree = {best_attribute:{}}
            attrs = [i for i in attrs if i != best_attribute]
            Y = pd.concat([y,X], axis=1)
            for value in np.unique(Y[best_attribute]):
                value= float(value)
                sub_tree = Y.where(Y[best_attribute]==value).dropna()
                sub_tree1 = sub_tree[sub_tree.columns[1:]]
                y = sub_tree['class']
                subtree = self.train(sub_tree1, y, attrs)
                tree[best_attribute][value] = subtree
        self.tree = tree
        return self.tree

    def predict(self, query, tree):
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.
        default = "X"
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]]
                except:
                    return default
                result = tree[key][query[key]]
                if isinstance(result,dict):
                    return self.predict(query, result)
                else:
                    return result

    def test(self, X, y, display=False):
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        Y = pd.concat([y,X], axis=1)
        x = list(X)
        X = X.copy()
        for i in range(len(x)):
            z = x[i]
            X[z] = X[z].astype('float')
        queries = X.to_dict(orient="records")
        predicted = pd.DataFrame(columns=["predicted"])
        y_true = Y["class"].values
        for i in range(len(queries)):
            predicted.loc[500+i,"predicted"] = self.predict(queries[i], self.tree)
        y_pred = predicted["predicted"].values
        recall = recall_score(y_true,y_pred, average='weighted')
        precision = precision_score(y_true,y_pred, average='weighted')
        accuracy = accuracy_score(y_true,y_pred)
        F1 = f1_score(y_true,y_pred, average='weighted')
        confusionmatrix = confusion_matrix(y_true, y_pred, labels=['B', 'L', 'R'])
        result = {'precision':precision,
                  'recall':recall,
                  'accuracy':accuracy,
                  'F1':F1,
                  'confusionmatrix':confusion_matrix}
        if display:
            print(result)

        return result
    
    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        try:
            return str(self.tree)
        except:
            return "ID3 untrained"

    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pickle.dump(self.tree, output)
        output.close()


