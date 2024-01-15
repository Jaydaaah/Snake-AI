from SnakeGame import *
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random as ran
import pygame

import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

from stable_baselines3 import PPO


class GameEnv(Env, GameStatusCallback):
    Truncated = False
    Done = False
    SnakeGame = None
    direction = 0
    reward = 0
    left_side_direction = [2, 3, 1, 0]
    right_side_direction = [3, 2, 0, 1]
    
    def __init__(self, snake_game = SnakeGame) -> None:
        super().__init__()
        
        self.SnakeGame = snake_game
        self.direction = snake_game.direction
        snake_fov_shape = self.SnakeGame.SNAKE_FOV.VISION_SHAPE[0]
        
        # set callback
        snake_game.set_callback(self)
        
        self.observation_space = Box(low=-1, high=3, shape=(snake_fov_shape * snake_fov_shape,), dtype=int)
        self.action_space = Discrete(3) # 0 - no action, 1 - left side of the snake head, 2 - right side of the snake head
    
    def StatusUpdate(self, status_update: int, **kwargs) -> None:
        match (status_update):
            case self.status_killed:
                self.reward -= 10
                self.Done = True
            case self.status_resurrected:
                self.direction = self.SnakeGame.direction
            case self.status_eat:
                self.reward += 5
            case self.status_near_food:
                self.reward += 1
            case self.status_change_direction:
                self.direction = self.SnakeGame.direction
            case self.status_quit:
                self.Truncated = True
        
    def step(self, action: int):
        snake_game = self.SnakeGame
        direction = self.direction
        
        if (action == 1):
            direction = self.left_side_direction[direction]
        elif (action == 2):
            direction = self.right_side_direction[direction]
                
        snake_game.change_direction(direction)
        
        obs = self._get_obs()
        
        try:
            return obs, self.reward, self.Done, self.Truncated, {}
        finally:
            self.Done = False
            self.reward = 0
            self.render()
    
    def render(self):
        if self.SnakeGame.GAME_RUNNING:
            self.SnakeGame.update()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(1)
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        return snake_game.get_flatten_observation()
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            name = 'best_model_{}'.format(self.n_calls)
            model_path = os.path.join(self.save_path, name)
            print(f"Save model: {name}")
            self.model.save(model_path)
        return True
    
from threading import Thread

def start_learning(model: PPO):
    model.learn(total_timesteps=100000, callback=callback)
    
if __name__ == '__main__':
    pygame.init()
    snake_game = SnakeGame()
    env = GameEnv(snake_game)
    
    env_checker.check_env(env)
    CHECKPOINT_DIR = './train/'
    LOG_DIR = "./logs/"
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    print("start learning....")
    model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=1)
    #model = PPO.load(r"train\best_model_1.zip", env)
    t = Thread(target = start_learning, args=(model,))
    t.start()
    #model.learn(total_timesteps=100000, callback=callback)
    print("done learning....")
    """
    for episode in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
        print(f"Total Reward for episode {episode} is {total_reward}")
        time.sleep(1)
    """
    pygame.quit()

