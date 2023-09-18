import numpy as np
from src.SumTree import SumTree

class PrioritisedMemory:
    # some cheeky hyperparameters
    eps = 0.01 # Epsilon - minimum priority possible
    a = 0.06 # Alpha - how much prioritization is used - a = 0 is uniform
    b = 0.04 # Beta - amount of importance sampling correction
    bIncreaseRate = 0.001
    errorsClippedAt = 1.0 # Highest priority possible

    def __init__(self, capacity, input_shape, n_actions, discrete=False):
        self.sumTree = SumTree(capacity)

        self.eps = 0.01 # Epsilon - minimum priority possible
        self.a = 0.06 # Alpha - how much prioritization is used - a = 0 is uniform
        self.b = 0.04 # Beta - amount of importance sampling correction
        self.bIncreaseRate = 0.001
        self.errorsClippedAt = 1.0 # Highest priority possible
        self.discrete = discrete
        self.max_priority = self.eps
        self.mem_size = capacity

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

        self.mem_count = 0 # Current memory count
        self.real_size = 0 # No of entries actually in buffer

    def store_transition(self, state, action, reward, state_, done):
        """ when an experience is first added to memory it has the highest priority
            so each experience is run through at least once
        """
        index = self.mem_count % self.mem_size # Current posn in memory

        # store experience index with maximum priority in sum tree
        self.sumTree.add(self.max_priority, index)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            # All zero except action taken
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        self.mem_count += 1
        self.read_size = min(self.mem_size, self.real_size + 1) # Needed?
        return


    def sample_buffer(self, batch_size):
        batchIndexes = []
        sampleIndexes = []
        batchISWeights = np.zeros([batch_size], dtype=np.float32)

        # so we divide the priority space up into n different priority segments
        totalPriority = self.sumTree.total_priority()
        prioritySegmentSize = totalPriority / batch_size

        # also we need to increase b with every value to anneal it towards 1
        self.b += self.bIncreaseRate
        self.b = min(self.b, 1)

        # in order to normalize all the weights in order to ensure they are all within 0 and 1
        # we are going to need to get the maximum weight and divide all weights by that

        # the largest weight will have the lowest priority and thus the lowest probability of being chosen
        #minPriority = np.min(np.maximum(self.sumTree.tree[self.sumTree.indexOfFirstData:], self.eps))
        minPriority = np.min(self.sumTree.tree[self.sumTree.indexOfFirstData:])
        minPriority = max(minPriority, self.eps) # Cannot go lower than epsilon
        
        minProbability = minPriority / totalPriority
        maxWeight = (minProbability * batch_size) ** (-self.b)

        for i in range(batch_size):
            # get the upper and lower bounds of the segment
            segmentMin = prioritySegmentSize * i
            segmentMax = segmentMin + prioritySegmentSize # e.g. segment*(i+1)

            value = np.random.uniform(segmentMin, segmentMax)

            treeIndex, priority, sampleIndex = self.sumTree.getLeaf(value)

            #batchIndexes[i] = treeIndex
            batchIndexes.append(treeIndex)
            #sampleIndexes[i] = sampleIndex
            sampleIndexes.append(sampleIndex)

            samplingProbability = priority / totalPriority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi

            batchISWeights[i] = np.power(batch_size * samplingProbability, -self.b) / maxWeight

        states = self.state_memory[sampleIndexes]
        new_states = self.new_state_memory[sampleIndexes]
        actions = self.action_memory[sampleIndexes]
        rewards = self.reward_memory[sampleIndexes]
        terminal = self.terminal_memory[sampleIndexes]

        return batchIndexes, states, actions, rewards, new_states, terminal, batchISWeights

        #return batchIndexes, batch, batchISWeights

    def batchUpdate(self, treeIndexes, absoluteErrors):
        """
        Update the priorities for the batch that has just been used, using
        the absoluteError used for training
        """
        #absoluteErrors += self.eps  # do this to avoid 0 values
        clippedErrors = np.minimum(absoluteErrors + self.eps, self.errorsClippedAt)
        priorities = np.power(clippedErrors, self.a)

        for treeIndex, priority in zip(treeIndexes, priorities):
            self.sumTree.update(treeIndex, priority)

            self.max_priority = max(self.max_priority, priority)
            
        return

    def save_buffer(self, filename):
        """
        Save contents of the memory buffer to external files.
        """
        np.save(filename + '_state', self.state_memory)
        np.save(filename + '_new_state', self.new_state_memory)
        np.save(filename + '_action', self.action_memory)
        np.save(filename + '_reward', self.reward_memory)
        np.save(filename + '_terminal', self.terminal_memory)

        treePointer = self.sumTree.save_tree(filename)
        return treePointer

    def load_buffer(self, filename, treePointer):
        """
        Load contents of the memory buffer from external files.
        """
        self.state_memory = np.load(filename + '_state.npy')
        self.new_state_memory = np.load(filename + '_new_state.npy')
        self.action_memory = np.load(filename + '_action.npy')
        self.reward_memory = np.load(filename + '_reward.npy')
        self.terminal_memory = np.load(filename + '_terminal.npy')
        self.sumTree.load_tree(filename, treePointer)
        return


class ExperienceBuffer:
    """
    This class will be used to store the past experiences of the environment (up to max_size)
    for the training of the DDQN model.
    """
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_count = 0
        self.discrete = discrete

        # Memory Buffers - could also use deque instead of numpy arrays
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, done):
        """
        Store the latest experience of the model (for later use)
        """
        index = self.mem_count % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            # All zero except action taken
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_count += 1

    def sample_buffer(self, batch_size):
        """
        Return a random sample of `batch_size` previous experiences for
        training.
        """
        max_mem = min(self.mem_count, self.mem_size) # Avoid sampling empty entries
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
    
    def save_buffer(self,filename):
        """
        Save contents of the memory buffer to external files.
        """
        np.save(filename + '_state', self.state_memory)
        np.save(filename + '_new_state', self.new_state_memory)
        np.save(filename + '_action', self.action_memory)
        np.save(filename + '_reward', self.reward_memory)
        np.save(filename + '_terminal', self.terminal_memory)

    def load_buffer(self,filename):
        """
        Load contents of the memory buffer from external files.
        """
        self.state_memory = np.load(filename + '_state.npy')
        self.new_state_memory = np.load(filename + '_new_state.npy')
        self.action_memory = np.load(filename + '_action.npy')
        self.reward_memory = np.load(filename + '_reward.npy')
        self.terminal_memory = np.load(filename + '_terminal.npy')