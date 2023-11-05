import numpy as np
import pygame
import pygame.surfarray as surfarray
from random import randint

import gymnasium as gym
from gymnasium import spaces


maze = [
    ['.', '', '.', '.'],
    ['.', '.', '.', 'O'],
    ['.', '.', '.', '.'],
    ['B', '.', '.', 'C'],
]


class JetPackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, hall, render_mode=None):
        self.hall = np.array(hall)
        # argwhere()[0] finds where elements are located in grid and returns coordinates in [x,y] shape
        self.start_pos = np.argwhere(self.hall == "B")[0]
        self.obstacle_pos = np.argwhere(self.hall == "O")[0]
        self.coin_pos = np.argwhere(self.hall == "C")[0]
        self.current_barry_pos = self.start_pos

        # 0= jetpack off 1 = jetpack on
        self.action_space = spaces.Discrete(2)

        # initially only one coin and obstacle eventually more
        self.observation_space = spaces.Dict(
            {
                "barry": spaces.Box(0, len(self.hall)-1, shape=(2,), dtype=int),
                "coin": spaces.Box(0, len(self.hall)-1, shape=(2,), dtype=int),
                # obstacle should actually take up 3 units
                "obstacle": spaces.Box(0, len(self.hall)-1, shape=(2,), dtype=int)
            }
        )

        pygame.init()
        self.cell_size = 125
        self.screen = pygame.display.set_mode(
            (len(self.hall[0]) * self.cell_size, len(self.hall) * self.cell_size))

        # code from gym tutorial
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # seed random
        super().reset(seed=seed)
        # return barry to start / floor
        self.current_barry_pos = self.start_pos

        # calc random vals for coin and obstacle

        # obs must be calc last
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        new_barry_pos = np.array(self.current_barry_pos)
        if (action == 1):
            new_barry_pos[0] -= 1
        if (action == 0):
            new_barry_pos[0] += 1

        if self._is_valid_position(new_barry_pos):
            self.current_barry_pos = new_barry_pos

        self._update_coin_postion()
        self._update_obstacle_position()

        reward, done = self._calc_reward()
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            print("calling frame")
            return self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        self.screen.fill((128, 128, 128))
        for row in range(len(self.hall)):
            for col in range(len(self.hall[0])):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                # draw background
                if (self.hall[row, col] == '.'):
                    pygame.draw.rect(
                        self.screen, (128, 128, 128), (cell_left, cell_top, self.cell_size, self.cell_size))
                # draw coin
                if np.array_equal([row, col], self.coin_pos):
                    pygame.draw.rect(
                        self.screen, (255, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                # draw obstacle
                elif np.array_equal([row, col], self.obstacle_pos):
                    pygame.draw.rect(
                        self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                # draw berry
                if (np.array_equal([row, col], self.current_barry_pos)):
                    pygame.draw.rect(
                        self.screen, (210, 180, 140), (cell_left, cell_top, self.cell_size, self.cell_size))

        if (self.render_mode == "human"):
            pygame.display.update()

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _calc_reward(self):
        reward = 0
        done = False
        if (np.array_equiv(self.current_barry_pos, self.coin_pos)):
            reward = 1
        if (np.array_equiv(self.current_barry_pos, self.obstacle_pos)):
            reward = -5
            done = True
        return reward, done

    def _update_coin_postion(self):
        self.coin_pos[1] -= 1
        if (self.coin_pos[1] < 0):
            self.coin_pos[1] = len(self.hall[1])
            self.coin_pos[0] = randint(0, len(self.hall)-1)

    def _update_obstacle_position(self):
        self.obstacle_pos[1] -= 1
        if (self.obstacle_pos[1] < 0):
            self.obstacle_pos[1] = len(self.hall[1])
            self.obstacle_pos[0] = randint(0, len(self.hall)-1)

    def _is_valid_position(self, postition):
        if (postition[0] > len(self.hall)-1 or postition[0] < 0):
            return False
        if (postition[1] > len(self.hall[0]) - 1 or postition[1] < 0):
            return False
        return True

    def _get_obs(self):
        return {"barry": self.current_barry_pos, "coin": self.coin_pos, "obstacle": self.obstacle_pos}
