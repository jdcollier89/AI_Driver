import pygame
import numpy as np

from src.ddqn import DDQNAgent
from src.Game import Game

# run = True
# game = Game()
# FPS = 60
# clock = pygame.time.Clock()


# while run:
#     clock.tick(FPS)
#     game.draw()

#     run = game.manual_loop()
#     model_input = game.game_state()
#     pygame.display.update()

# pygame.quit()

train_model = True

if __name__ == '__main__':
    game = Game()

    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=9, epsilon=1.0, batch_size=32, input_dims=10)
    # Should up to 12 input dims --> angle to reward gate + drift mom

    if train_model:
        # Train Model
        current_ep = 0
        n_games = 100
        max_steps = 3600
        #ddqn_agent.load_model()
        ddqn_scores = []
        eps_history = []

        while current_ep <= n_games:
            score = 0
            lifespan_ = 0
            game.game_reset()
            game_state, reward, done = game.game_state()
            steps = 0

            while steps < max_steps:
                lifespan = lifespan_
                lifespan_ += 1
                action = ddqn_agent.choose_action(game_state)
                #action = action + 1 # Account for index offset
                game.game_loop(action+1)
                game_state_, reward, done = game.game_state()
                score += reward
                ddqn_agent.remember(game_state, action, reward, game_state_, done)
                game_state = game_state_
                ddqn_agent.train()
                if done: # End episode if car crashed
                    steps = max_steps
                if steps % 100 == 0:
                    print(f"{steps} steps done!")
                steps += 1

            eps_history.append(ddqn_agent.epsilon) # Look at training steps instead?
            ddqn_scores.append(score)

            print('Episode no ', current_ep, 'score %.2f' % score, 'lifespan ', lifespan)
                
            if current_ep % 1 == 0 and current_ep > 0:
                ddqn_agent.save_model()
                print(f"Saving model after {ddqn_agent.epsilon_step} - episode {current_ep}")

            current_ep += 1
    else:
        # Test model
        game.game_reset()
        game_state, reward, done = game.game_state()
        run = True
        FPS = 60
        clock = pygame.time.Clock()
        ddqn_agent.load_model()
        while run:
            clock.tick(FPS)
            game.draw()
            action = ddqn_agent.choose_action(game_state)
            game.game_loop(action+1)
            game_state, reward, done = game.game_state()
            if done:
                run = False
            pygame.display.update()
        pygame.quit()