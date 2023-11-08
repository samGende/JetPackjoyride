import numpy as np
import pygame
import pygame.surfarray as surfarray
from random import randint

import gymnasium as gym
from gymnasium import spaces


maze = [
    ['.', '.', '.', '.'],
    ['.', '.', '.', 'O'],
    ['.', '.', '.', '.'],
    ['B', '.', '.', 'C'],
]


class JetPackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.hall = np.zeros((64, 64))
        self.start_pos = [len(self.hall)-1, 0]
        self.current_barry_pos = self.start_pos
        # list of obstacles and coins first in the list will be observed
        self.coins = [[self.np_random.integers(
            0, len(self.hall)),  len(self.hall[1])]]
        self.obstacles = [self._gen_obstacle()]
        self.obstacle_pos = self.obstacles[0]
        self.coin_pos = self.coins[0]
        self.coin_pos = self.coins[0]
        self.score = 0
        self.obstacle_pos = self.obstacles[0]

        # 0= jetpack off 1 = jetpack on
        self.action_space = spaces.Discrete(2)

        low = np.array([0, 0])  # Minimum coordinates in each dimension
        # Maximum coordinates in each dimension
        high = np.array([len(self.hall)-1, len(self.hall)-1])
        obstacle_low = np.array([low, low, low])
        obstacle_high = np.array(
            [high, [high[0], high[1] + 1], [high[0], high[1] + 2]])

        # initially only one coin and obstacle eventually more
        self.observation_space = spaces.Dict(
            {
                "barry": spaces.Box(low, high, dtype=int),
                "coin": spaces.Box(low,  high, dtype=int),
                # obstacle should actually take up 3 units
                "obstacle": spaces.Box(obstacle_low, obstacle_high, shape=(3, 2), dtype=int)
            }
        )

        pygame.init()
        self.cell_size = 8
        self.screen = pygame.display.set_mode(
            (len(self.hall[0]) * self.cell_size, len(self.hall) * self.cell_size))

        self.font = pygame.font.SysFont(None, 48)
        self.img = self.font.render(f'Score {self.score}', True, (255, 0, 0))

        original_image = pygame.image.load(
            'Barry_Steakfries.webp').convert_alpha()
        original_coin_image = pygame.image.load(
            'Coin2.webp').convert_alpha()

        self.coin_sprite = pygame.transform.scale(
            original_coin_image, (24, 24))
        self.sprite_sheet_image = pygame.transform.scale(
            original_image, (24, 24))

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
        self.coins = [[self.np_random.integers(
            0, len(self.hall)),  len(self.hall[1]) - 1]]
        self.obstacles = [self._gen_obstacle()]
        self.obstacle_pos = self.obstacles[0]
        self.coin_pos = self.coins[0]

        self.score = 0
        self.img = self.font.render(f'Score {self.score}', True, (255, 0, 0))

        # calc random vals for coin and obstacle

        # obs must be calc last
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        new_barry_pos = np.array(self.current_barry_pos)
        # jetpack not on
        if (action == 1):
            new_barry_pos[0] -= 1
        # jetpack is on
        if (action == 0):
            new_barry_pos[0] += 1
        # check if barry will fly of the screen
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
            return self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        self.screen.fill((128, 128, 128))
        self.screen.blit(self.img, (20, 20))

        # draw coins
        for coin in self.coins:
            self.screen.blit(self.coin_sprite,
                             ((coin[1] - 1)*self.cell_size, (coin[0]-1) * self.cell_size))

        # draw obstacles
        for obstacle in self.obstacles:
            for block in obstacle:
                pygame.draw.rect(self.screen, (0, 0, 0), (
                    block[1] * self.cell_size, block[0] * self.cell_size, self.cell_size, self.cell_size))

        # draw berry w sprite
        self.screen.blit(self.sprite_sheet_image,
                         ((self.current_barry_pos[1] - 1)*self.cell_size, (self.current_barry_pos[0]-1) * self.cell_size))
        if (self.render_mode == "human"):
            pygame.display.update()

        return np.transpose(
            # return a rgb array
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _calc_reward(self):
        reward = 0
        done = False
        if (np.array_equiv(self.current_barry_pos, self.coin_pos)):
            reward = 1
            self._update_score()

        for block in self.obstacle_pos:
            if (np.array_equiv(self.current_barry_pos, block)):
                reward = -5
                done = True
        return reward, done

    def _update_score(self):
        self.score += 1
        self.img = self.font.render(f'Score {self.score}', True, (255, 0, 0))

    def _update_coin_postion(self):
        for coin in self.coins:
            coin[1] -= 1
        self.coin_pos = self.coins[0]
        if (self.coin_pos[1] < 0):
            self.coins.pop(0)
            self.coins.append(
                [self.np_random.integers(0, len(self.hall)), len(self.hall[1])])
            self.coin_pos = self.coins[0]
        if (len(self.coins) < 10):
            self._gen_new_coin()

    def _gen_new_coin(self):
        newCoin = randint(1, 4)
        if (newCoin == 1):
            self.coins.append(
                [self.np_random.integers(0, len(self.hall)), len(self.hall[1])])

    def _update_obstacle_position(self):
        for obstacle in self.obstacles:
            for block in obstacle:
                block[1] -= 1
        self.obstacle_pos = self.obstacles[0]
        if (self.obstacle_pos[2][1] < 0):
            self.obstacles.pop(0)
            self.obstacles.append(self._gen_obstacle())
            self.obstacle_pos = self.obstacles[0]
        if (len(self.obstacles) < 10):
            self._generate_obstacles()

    def _generate_obstacles(self):
        newObstacle = randint(1, 4)
        if (newObstacle == 1):
            self.obstacles.append(
                self._gen_obstacle())

    def _is_valid_position(self, postition):
        if (postition[0] > len(self.hall)-1 or postition[0] < 0):
            return False
        if (postition[1] > len(self.hall[0]) - 1 or postition[1] < 0):
            return False
        return True

    def _get_obs(self):
        obs = {"barry": np.array(self.current_barry_pos, dtype=int),
               "coin": np.array(self.coin_pos, dtype=int),
               "obstacle": np.array(self.obstacle_pos, dtype=int)}
        return obs

    def _gen_obstacle(self):
        type = self.np_random.integers(0, 4)
        # list of 3 coordinates of the obstacle
        obstacle = []
        if (type == 0):
            # the rightmost part of the obstacle
            obstacle.append(
                [self.np_random.integers(0, len(self.hall)), len(self.hall[1])-1])
            obstacle.append([obstacle[0][0], obstacle[0][1] + 1])
            obstacle.append([obstacle[0][0], obstacle[0][1] + 2])
        elif (type == 1):
            obstacle.append(
                [self.np_random.integers(1, len(self.hall)-1), len(self.hall[1])-1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]])
            obstacle.append([obstacle[0][0]+1, obstacle[0][1]])
        elif (type == 2):
            obstacle.append(
                [self.np_random.integers(1, len(self.hall)-1), len(self.hall[1])-1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]+1])
            obstacle.append([obstacle[0][0]+1, obstacle[0][1]-1])
        elif (type == 3):
            obstacle.append(
                [self.np_random.integers(1, len(self.hall)-1), len(self.hall[1])-1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]-1])
            obstacle.append([obstacle[0][0]+1, obstacle[0][1]+1])

        return obstacle
