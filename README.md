# Self-driving car in 2D

This creates an agent which uses Double Deep Q-learning (DDQN) [1] with Prioritized Experience Replay (PER) [2] to learn to navigate around a course in a 2D environment.

## Setup/Installation

## Commands

The Python package **Click** is used to create a command line interface for running the game. This allows the user to directly run different functions from the `main.py` without having to change anything within the code.

Run the code in this repository with 

``python main.py {click-command}``

for example, to further train a model you would use the command `python main.py train`.

- **train**: Load the model saved in the `model` subdirectory and continue the process of training. The will be loaded in with the model. No visuals will be shown while the model trains (in order to save resources).

- **test**: Load the model saved in the `model` subdirectory and use it for automatic control of the car (PyGame screen will show the game in progress). 

- **manual**: Play the game yourself, with no AI involvement, and controlling the car with **W, A, S** and **D** keys. In manual mode, a collision with the barrier will not result in a gameover, instead the car will bounce off.

## How the reinforcement model works

### Rewards

As the agent requires some sort of reward system, I have used a series of checkpoints (or reward gates) at approximately equal distance along the track. The agent will receive a reward when passing through the active checkpoint, and will have input parameters giving the distance to the next reward gate and relative angle to the next gate (taking the direction the car is facing as 0 degrees).

- Car Crashes: -100

- Car passes a checkpoint (reward gate): 25

- Car has velocity less than 0.1: -5 per tick

- Car has velocity less than 1 (but more than 0.1): -3 per tick

- Car has velocity of higher than 1: -1 per tick

### Computer vision vectors

### Current Performance

## Possible extensions in the future

- Normalize input vector

- Optimize performance of PyGame component.

- Try Dueling Deep Q-learning network


## References

1. Hado van Hasselt, Arthur Guez and David Silver (2015). Deep Reinforcement Learning with Double Q-learning. ArXiv:1509.06461. https://arxiv.org/abs/1509.06461

2. Tom Schaul, John Quan and Ioannis Antonoglou and David Silver (2016). Prioritized Experience Replay. ArXiv:1511.05952. http://arxiv.org/abs/1511.05952
