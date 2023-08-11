from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

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
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

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
    
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    """
    lr = learning rate
    n_actions = no. of available actions in environment
    input_dims = no. of inputs given to model from environment
    fc1_dims = Dims. of first fully connected layer
    fc2_dims = Dims. of second fully connected layer
    """
    model = Sequential([
                Dense(fc1_dims, activation='relu', input_shape=[input_dims, ], name="fc_layer1"),
                Dense(fc2_dims, activation='relu', name="fc_layer2"),
                Dense(n_actions, name="output_layer")
            ])
        
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


class DDQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.00001, epsilon_end=0.01,
                 mem_size=100000, fname='ddqn_model.h5', replace_target=1000,
                 parameter_fname = 'ddqn_model'):
        """replace_target = how often to update the target model"""
        
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)] # e.g. [0, 1, 2, 3]
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.epsilon_step = 1

        self.batch_size = batch_size
        self.model_file = fname
        self.param_fname = parameter_fname
        self.replace_target = replace_target
        self.memory = ExperienceBuffer(mem_size, input_dims, n_actions, True)

        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = np.array(state).astype(np.float32)
        state = state[np.newaxis,:]
        # Either explore or follow policy
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        # Return index of chosen action - start from 0
        return action
    
    def train(self):
        # Instead of filling memory buffer with random input, don't train until buffer full
        if self.memory.mem_count > self.batch_size:
            state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state, verbose=0) # What reward would target model expect for each poss. action
            q_eval = self.q_eval.predict(new_state, verbose=0)

            q_pred = self.q_eval.predict(state, verbose=0) # Reward of each action that could have been taken

            max_actions = np.argmax(q_eval, axis=1) # What action would the live model suggest

            q_target = q_pred # To ensure diff only found for actions actually taken

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                self.gamma * q_next[batch_index, max_actions.astype(int)] * done
            
            _ = self.q_eval.fit(state, q_target, verbose=0)

            # self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
            #                 self.epsilon_min else self.epsilon_min

            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
                -self.epsilon_dec * self.epsilon_step)
            
            self.epsilon_step += 1
            if self.memory.mem_count % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        """
        Update QTarget with the weights of QEval model
        """
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self, ep_no):
        self.q_eval.save(self.model_file)
        self.memory.save_buffer(self.param_fname)

        steps = np.array([ep_no, self.epsilon_step, self.memory.mem_count])
        np.save(self.param_fname + '_steps', steps)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        self.memory.load_buffer(self.param_fname)

        steps = np.load(self.param_fname + '_steps.npy')
        ep_no = steps[0] + 1 # Increment as we are starting a new episode
        self.epsilon_step = steps[1]
        self.memory.mem_count = steps[2]

        if self.epsilon >= self.epsilon_min:
            self.update_network_parameters()

        return ep_no