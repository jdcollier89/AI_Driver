import pygame
import click
import numpy as np
import gc

from src.ddqn import DDQNAgent
from src.Game import Game

FPS = 30

@click.group()
def cli():
    pass


@cli.command()
def manual():
    game = Game()
    clock = pygame.time.Clock()

    game.draw()
    pygame.display.update()

    run = True
    while run:
        clock.tick(FPS)
                
        #game.draw()
        run = game.manual_loop()
        _ = game.game_state()

        game.draw()
        pygame.display.update()

    pygame.quit()


@cli.command()
def train():
    game = Game()
    pygame.event.set_allowed([pygame.QUIT])

    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.95, n_actions=9, epsilon=1.0, batch_size=64, input_dims=12, 
                           fname='model/ddqn_model.h5', parameter_fname = 'model/ddqn_model')

    # Train Model
    current_ep = 0
    n_games = 10000
    max_steps = 3600
    #current_ep = ddqn_agent.load_model()

    while current_ep <= n_games:
        score = 0
        lifespan_ = 0
        game.game_reset()
        game_state, reward, done = game.game_state()
        steps = 0

        while steps < max_steps:
            lifespan = lifespan_
            lifespan_ += 1
            action = ddqn_agent.choose_action_train(game_state)
            game.game_loop(action+1)
            game_state_, reward, done = game.game_state()
            score += reward
            ddqn_agent.remember(game_state, action, reward, game_state_, done)
            game_state = game_state_
            ddqn_agent.train()
            running = game.check_exit()
            if done or not(running): # End episode if car crashed
                steps = max_steps
            steps += 1

        if not(running):
            break
        
        print(f'Episode finished with {game.gate_count} reward gates passed.')
        print('Episode no ', current_ep, 'score %.2f' % score, 'lifespan ', lifespan)
        
        gc.collect()
            
        if current_ep % 25 == 0:
            ddqn_agent.save_model(current_ep)
            print(f"Saved model after {ddqn_agent.epsilon_step} training steps <- episode {current_ep}")

        current_ep += 1

    pygame.quit()


@cli.command()
def test():
    # Test model
    game = Game()
    pygame.event.set_allowed([pygame.QUIT])

    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.95, n_actions=9, epsilon=1.0, batch_size=64, input_dims=12, 
                           fname='model/ddqn_model.h5', parameter_fname = 'model/ddqn_model')
    game.game_reset()
    game_state, _, done = game.game_state()

    clock = pygame.time.Clock()
    _ = ddqn_agent.load_model()
    game.draw()
    pygame.display.update()

    run = True
    while run:
        clock.tick(FPS)
        action = ddqn_agent.choose_action(game_state)
        game.game_loop(action+1)
        game_state, _, done = game.game_state()
        run = game.check_exit()
        if done: # End episode if car crashed
            print(f'Attempt finished with {game.gate_count} reward gates passed.')
            game.game_reset()
            game_state, _, done = game.game_state()
        game.draw()
        pygame.display.update()
    pygame.quit()


@cli.command()
def record():
    # Record the steps that are taken by a model for replay later
    game = Game()
    pygame.event.set_allowed([pygame.QUIT])

    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.95, n_actions=9, epsilon=1.0, batch_size=64, input_dims=12, 
                           fname='model/ddqn_model.h5', parameter_fname = 'model/ddqn_model')
    game.game_reset()
    game_state, _, done = game.game_state()

    clock = pygame.time.Clock()
    _ = ddqn_agent.load_model()
    steps = 0
    max_steps = 3600
    actions = []
    game.draw()
    pygame.display.update()

    run = True
    while run:
        clock.tick(FPS)
        action = ddqn_agent.choose_action(game_state)
        actions.append(action)
        game.game_loop(action+1)
        game_state, _, done = game.game_state()
        steps += 1
        
        game.draw()
        pygame.display.update()

        run = game.check_exit()
        if done or (steps > max_steps): # End episode if car crashed
            print(f'Attempt finished with {game.gate_count} reward gates passed, after {steps} steps.')
            break
    np.save('model/action_save', actions)
    pygame.quit()


@cli.command()
def playback():
    # Playback a pre-recorded game
    game = Game()
    pygame.event.set_allowed([pygame.QUIT])

    game.game_reset()
    _ = game.game_state()

    clock = pygame.time.Clock()

    steps = 0
    actions = np.load('model/action_save.npy')
    game.draw()
    pygame.display.update()

    run = True
    while run:
        clock.tick(FPS)
        action = actions[steps]
        game.game_loop(action+1)
        _ = game.game_state()
        steps += 1
        game.draw()
        pygame.display.update()
        run = game.check_exit()
        if steps == len(actions):
            print(f'Attempt finished with {game.gate_count} reward gates passed, after {steps} steps.')
            break
    pygame.quit()


if __name__ == '__main__':
    cli()