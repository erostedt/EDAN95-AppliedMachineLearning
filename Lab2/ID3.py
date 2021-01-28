from collections import Counter
from graphviz import Digraph
import numpy as np
import math


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes):
        size = len(data)
        attribute_counter = {}
        counter = 0

        attribute2index = {}
        counter = 0
        for a in attributes.keys():
            attribute2index[a] = counter
            counter += 1

        for key, value in attributes.items():
            attribute_counter[key] = {}
            for a in value:
                attribute_counter[key][a] = {}
                attribute_counter[key][a]["total"] = 0
                attribute_counter[key][a]["+"] = 0
                attribute_counter[key][a]["-"] = 0

        for i in range(len(data)):
            for att in data[i]:
                for dict_values in attribute_counter.values():
                    if att in dict_values:
                        dict_values[att]["total"] += 1
                        if target[i] == "+":
                            dict_values[att]["+"] += 1
                        else:
                            dict_values[att]["-"] += 1

        n_pos = 0
        n_neg = 0
        total_entropy = 0
        for a in target:
            if a == "+":
                n_pos += 1
            else:
                n_neg += 1

        total_entropy -= (n_pos / size) * math.log2(n_pos / size) if n_pos != 0 else 0
        total_entropy -= (n_neg / size) * math.log2(n_neg / size) if n_neg != 0 else 0

        print(total_entropy)

        entropies = {}
        for attribute in attributes.keys():
            entropies[attribute] = {}
            for label in attribute_counter[attribute].keys():
                entropies[attribute][label] = self.entropy(attribute_counter[attribute][label])
        print(attribute_counter)
        print(entropies)

        attribute_to_split = ""
        last_max = 0
        for key, value in entropies.items():
            information_gain = total_entropy
            for k, v in value.items():
                information_gain -= (attribute_counter[key][k]["total"] / size) * v
            if information_gain > last_max:
                last_max = information_gain
                attribute_to_split = key
        node = self.new_ID3_node()
        node["attribute"] = attribute_to_split
        node["entropy"] = total_entropy
        node["samples"] = size
        node["classCounts"] = {"+": n_pos, "-": n_neg}

        temp_data = [[], []]
        temp_target = [[], []]
        for i in range(len(data)):
            if data[i][attribute2index[attribute_to_split]] == attributes[attribute_to_split][0]:
                temp_data[0].append(data[i])
                temp_target[0].append(target[i])
            else:
                temp_data[1].append(data[i])
                temp_target[1].append(target[i])

        del attributes[attribute_to_split]
        return node, temp_data, temp_target, attributes

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        node, data, target, attributes = self.find_split_attr(data, target, attributes)
        root = node
        parent_id = node["id"]
        self.add_node_to_graph(root)
        for i in range(len(data)):
            node, data, target, attributes = self.find_split_attr(data[i], target[i], attributes)
            self.add_node_to_graph(node, parent_id)

        # root = node
        # self.add_node_to_graph(root)
        # fill in something more sensible here... root should become the output of the recursive tree creation
        # root = self.new_ID3_node()
        # self.add_node_to_graph(root)

        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted

    @staticmethod
    def entropy(dict):
        res = 0
        total = dict["total"]
        pos = dict["+"]
        neg = dict["-"]
        res -= (pos / total) * math.log2(pos / total) if pos != 0 else 0
        res -= (neg / total) * math.log2(neg / total) if neg != 0 else 0
        return res
