# Self-driving car in 2D

This creates an agent which uses Double Deep Q-learning (DDQN) [1] with prioritized experience replay (PER) [2] to learn to navigate around a course in a 2D environment.

## Setup/Installation

## Commands

The Python package **Click** is used to create a command line interface for running the game. This allows the user to directly run different functions from the `main.py` without having to change anything within the code.

Run the code in this repository with 

``python main.py {click-command}``

for example, to further train a model you would use the command `python main.py train`.

- **train**: Load the model saved in the `model` subdirectory and continue the process of training. The will be loaded in with the model. No visuals will be shown while the model trains (in order to save resources).

- **test**: Load the model saved in the `model` subdirectory and use it for automatic control of the car (PyGame screen will show the game in progress). 

- **manual**: Play the game yourself, with no AI involvement, and controlling the car with **W, A, S** and **D** keys.

## How the reinforcement model works

### Computer vision vectors

### Current Performance

## Possible extension in the future


## References

1. Hado van Hasselt, Arthur Guez and David Silver (2015).Deep Reinforcement Learning with Double Q-learning. ArXiv:1509.06461. https://arxiv.org/abs/1509.06461

2. Tom Schaul, John Quan and Ioannis Antonoglou and David Silver (2016). Prioritized Experience Replay. ArXiv:1511.05952. http://arxiv.org/abs/1511.05952