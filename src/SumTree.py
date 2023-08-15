import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 2 * capacity - 1
        self.tree = np.zeros(self.size)
        self.data = np.zeros(capacity, dtype=object)
        self.dataPointer = 0
        self.indexOfFirstData = capacity - 1

    """
    adds a new element to the sub tree (or overwrites an old one) and updates all effected nodes 
    """

    def add(self, priority, data):
        treeIndex = self.indexOfFirstData + self.dataPointer

        # overwrite data

        self.data[self.dataPointer] = data
        self.update(treeIndex, priority)
        self.dataPointer += 1
        self.dataPointer = self.dataPointer % self.capacity

    """
    updates the priority of the indexed leaf as well as updating the value of all effected
    elements in the sum tree
    """

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority

        while index != 0:
            # set index to parent
            index = (index - 1) // 2
            self.tree[index] += change

    def getLeaf(self, value):
        parent = 0
        LChild = 1
        RChild = 2

        while LChild < self.size:
            if self.tree[LChild] >= value:
                parent = LChild
            else:
                value -= self.tree[LChild]
                parent = RChild

            LChild = 2 * parent + 1
            RChild = 2 * parent + 2

        treeIndex = parent
        dataIndex = parent - self.indexOfFirstData

        return treeIndex, self.tree[treeIndex], self.data[dataIndex]

    def total_priority(self):
        return self.tree[0]  # Returns the root node
    
    def save_tree(self, filename):
        np.save(filename + '_tree', self.tree)
        np.save(filename + '_treeData', self.data, allow_pickle=True)

        return self.dataPointer
    
    def load_tree(self, filename, dataPointer):
        self.dataPointer = dataPointer

        self.tree = np.load(filename + '_tree.npy')
        self.data = np.load(filename + '_treeData.npy', allow_pickle=True)