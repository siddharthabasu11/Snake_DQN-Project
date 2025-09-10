import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame
import torch.nn.functional as S
import os
import pickle
# Game constants
GRID_SIZE = 15
SNAKE_START_POS = (8, 8)


class SnakeGame:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.snake_pos = [SNAKE_START_POS]
        self.direction = (1,0)
        self.apple_pos = self.apple_position()
        self.score = 0
        self.done = False
    
    def reset(self):
        self.snake_pos = [SNAKE_START_POS]
        self.direction = (1,0)
        self.apple_pos = self.apple_position()
        self.score = 0
        self.done = False
        return self.get_state()
    #what the agent sees at each step
    def get_state(self):
        head_x, head_y = self.snake_pos[0]
        apple_x, apple_y = self.apple_pos
        direction_x, direction_y = self.direction
        distance_x = apple_x - head_x
        distance_y = apple_y - head_y
        #normalize to keep dimensions in same range
        pos_x = head_x + direction_x
        pos_y = head_y + direction_y
        danger = (pos_x < 0 or pos_x >= self.grid_size or pos_y < 0 or pos_y >= self.grid_size
                  or (pos_x, pos_y) in self.snake_pos[1:])

        state = [head_x / self.grid_size ,
                 head_y / self.grid_size,
                 direction_x,
                 direction_y,
                 distance_x ,
                 distance_y ,
                 float(danger)]
        return np.array(state, dtype = np.float32)
    

    def step(self, action):
        if self.done:
            return self.get_state(), 0, self.done
        
        # different directions
        UP = (0, -1)
        RIGHT = (1, 0)
        DOWN = (0, 1)
        LEFT = (-1, 0)

        directions = [RIGHT, DOWN, LEFT, UP] 

        # Inside step()
        rel_index = directions.index(self.direction)
        if action == 0:  # straight
            new_direction = self.direction
        elif action == 1:  # turn left
            new_direction = directions[(rel_index - 1) % 4]
        elif action == 2:  # turn right
            new_direction = directions[(rel_index + 1) % 4]

        self.direction = new_direction
        
        reward = self.movement()
        if self.done:
            return self.get_state(), reward, self.done

        return self.get_state(), reward, self.done
        
    #move the snake in the current direction
    def movement(self):
        snake_pos_x, snake_pos_y = self.snake_pos[0]
        new_head_pos = (snake_pos_x + self.direction[0], snake_pos_y + self.direction[1])

        head_posx, head_posy = new_head_pos[0], new_head_pos[1]  

        #adding new reward to increase learning
        distance_initial = abs(snake_pos_x - self.apple_pos[0]) + abs(snake_pos_y - self.apple_pos[1])
        distance_new = abs(head_posx - self.apple_pos[0]) + abs(head_posy - self.apple_pos[1])

        if head_posx < 0 or head_posx >= self.grid_size or head_posy < 0 or head_posy >= self.grid_size or new_head_pos in self.snake_pos[1:]:
            self.done = True
            return -10
        
        self.snake_pos.insert(0, new_head_pos)
        if new_head_pos == self.apple_pos:
            self.score += 1
            self.apple_pos = self.apple_position()
            return 10
        total_reward = 0
        self.snake_pos.pop()
        total_reward += -0.01
        if distance_initial < distance_new:
            total_reward -=0.1
        else:
            total_reward += 0.1
        
        return total_reward
        


    def apple_position(self):
        while True:
            #spawn apples
            apple_pos = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if apple_pos not in self.snake_pos:
                return apple_pos
            
    def play_human(self):
        pygame.init()
        pixel = 30
        window = pixel * self.grid_size
        screen = pygame.display.set_mode((window, window))
        clock = pygame.time.Clock()

        self.reset()

        running = True
        MOVE_INTERVAL = 200  # ms between moves
        last_move = pygame.time.get_ticks()
        move_auto = False

        while running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    move_auto = True
                    if event.key == pygame.K_LEFT and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                        self.direction = (1, 0)
                    elif event.key == pygame.K_UP and self.direction != (0, 1):
                        self.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                        self.direction = (0, 1)

         
            now = pygame.time.get_ticks()
            if move_auto and now - last_move > MOVE_INTERVAL:
                if not self.done:
                    self.movement()
                last_move = now

            screen.fill((0, 0, 0))

            # draw snake
            for (x, y) in self.snake_pos:
                pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x * pixel, y * pixel, pixel, pixel))

            # draw apple
            apple_x, apple_y = self.apple_pos
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(apple_x * pixel, apple_y * pixel, pixel, pixel))

            # draw score
            font = pygame.font.SysFont('arial', 25)
            text = font.render("Score: " + str(self.score), True, (255, 255, 255))
            screen.blit(text, [5, 5])

            pygame.display.flip()
            clock.tick(60)  

        pygame.quit()
                

class DQN(nn.Module):
    def __init__(self, state_size = 7, action_size = 3):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(self, state_size = 7, action_size = 3):
        self.gamma = 0.9
        self.epsilon = 1
        self.maxlen = 10000
        self.memory = deque(maxlen = self.maxlen)
        self.q_network = DQN(state_size, action_size)
        #adding to run more episodes discontinuously

        self.target = DQN(state_size, action_size)
        self.target.load_state_dict(self.q_network.state_dict())
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.action_size = action_size
        if os.path.exists("snake_dqn.pth"):
            self.q_network.load_state_dict(torch.load("snake_dqn.pth"))
            self.target.load_state_dict(self.q_network.state_dict())
        
        if os.path.exists("replay.pkl"):
            with open("replay.pkl", "rb" )as f:
                self.memory = pickle.load(f)
        

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype = torch.float32)
        update_q = self.q_network(state_tensor)
        return torch.argmax(update_q).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, memory, batch_size):
        if(len(memory) > batch_size):
            mini = random.sample(memory, batch_size)
        else:
            mini = memory
        for state, action, reward, next_state, done in mini:
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype = torch.float32)
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype = torch.float32)
            q_values = self.q_network(state_tensor)
            q_chosen = q_values[0][action]
            target = torch.tensor(reward, dtype = torch.float32).squeeze()
            if not done:
                #DQN, when added DDQN worked better
                # next_q_values = self.target(next_state_tensor)
                # target = reward + self.gamma * torch.max(next_q_values).detach()
                next_q_values1 = self.q_network(next_state_tensor)
                action_1 = torch.argmax(next_q_values1, dim = 1)
                target_values1 = self.target(next_state_tensor)
                target = reward + self.gamma * target_values1[0][action_1].detach()

            #copy q_values, change q value of action with calculated target
            self.optimizer.zero_grad()
            #used mse-loss, upgrade to huber loss for more stability
            # loss = S.mse_loss(q_values, target_final)
            loss = S.smooth_l1_loss(q_chosen, target)
            loss.backward()
            self.optimizer.step()
            
    def train_short(self, state, action, reward, next_state, done):
        target = torch.tensor(reward, dtype = torch.float32).squeeze()
        #doesn't sample anymore
        state_tensor = torch.tensor(state.reshape((1,7)), dtype = torch.float32)
        next_state_tensor = torch.tensor(next_state.reshape((1,7)), dtype = torch.float32)
        q_values = self.q_network(state_tensor)

        q_chosen = q_values[0][action]
        if not done: 
            # next_q_values = self.target(next_state_tensor)
            # target = reward + self.gamma * (torch.max(next_q_values).detach())
            #DDQN
                next_q_values1 = self.q_network(next_state_tensor)
                action_1 = torch.argmax(next_q_values1, dim = 1)
                target_values1 = self.target(next_state_tensor)
                target = reward + self.gamma * target_values1[0][action_1].detach()
        self.optimizer.zero_grad()
        # loss = S.mse_loss(q_values, target_final)
        loss = S.smooth_l1_loss(q_chosen, target)
        loss.backward()
        self.optimizer.step()        
    
  
    def train_agent(self, game, episodes = 50000):
        scores = []
        for i in range(episodes):
            state = game.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                
                next_state, reward, done = game.step(action)

                self.remember(state, action, reward, next_state, done)

                self.train_short(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
            self.replay(self.memory, batch_size = 128)
            scores.append(game.score)

            if i%100 == 0:
                self.target.load_state_dict(self.q_network.state_dict())
                torch.save(self.q_network.state_dict(), "snake_dqn.pth")

                #implemented replay buffer storage so agent doesn't start new
                with open("replay.pkl", "wb" )as f:
                    pickle.dump(self.memory, f)
                

            if self.epsilon > 0.01:
                #tried, was too fast decay for # episodes
                # self.epsilon = max(0.01, self.epsilon * 0.999)
                self.epsilon = max(0.01, self.epsilon * 0.9997)


            if i%100 == 0:
                print(f"Episode {i}, Score: {game.score}, Epsilon: {self.epsilon:.3f}")
        torch.save(self.q_network.state_dict(), "snake_dqn.pth")
        return scores



def play_agent(agent, game):
    pygame.init()
    pixel = 30
    window = pixel * game.grid_size
    screen = pygame.display.set_mode((window, window))
    clock = pygame.time.Clock()

    state = game.reset()

    running = True
    MOVE_INTERVAL = 200  #tried different times, this was smooth between moves
    last_move = pygame.time.get_ticks()

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        now = pygame.time.get_ticks()
        if now - last_move > MOVE_INTERVAL and not game.done :
            action = agent.act(state)
            state, _, _ = game.step(action)
            last_move = now


        screen.fill((0, 0, 0))

        # draw snake
        for (x, y) in game.snake_pos:
            pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x * pixel, y * pixel, pixel, pixel))

        # draw apple
        apple_x, apple_y = game.apple_pos
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(apple_x * pixel, apple_y * pixel, pixel, pixel))

        # draw score
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(game.score), True, (255, 255, 255))
        screen.blit(text, [5, 5])

        pygame.display.flip()
        clock.tick(60)  

    pygame.quit()
if __name__ == "__main__":
    game = SnakeGame()
    # game.play_human()



    agent = DQNAgent()
    #added for more episodes without computer burning

    if os.path.exists("snake_dqn.pth"):
        agent.q_network.load_state_dict(torch.load("snake_dqn.pth"))
        agent.target.load_state_dict(agent.q_network.state_dict())
        agent.epsilon = 0.05

    scores = agent.train_agent(game, episodes = 15000)


    window = 200  # moving average window size
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    running_max = np.maximum.accumulate(scores)

    plt.figure(figsize=(12,6))
    plt.plot(scores, alpha=0.3, label="Raw scores")
    plt.plot(range(window-1, len(scores)), moving_avg, color="red", linewidth=2, label=f"{window}-episode moving avg")
    plt.plot(running_max, color="green", linewidth=2, label="Running max score")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Snake DQN Training Progress")
    plt.show()


    play_agent(agent, game)