import numpy as np
import pygame
import pygame.surfarray as surfarray
from random import randint

import gymnasium as gym
from gymnasium import spaces


class JetPackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.hall = np.zeros((64, 64))
        self.start_pos = [len(self.hall)-1, 0]
        self.current_barry_pos = self.start_pos
        # TODO change coins so that they are largers
        self.coins = [self.gen_coin()]
        self.obstacles = [self._gen_obstacle()]
        self.obstacle_pos = self.obstacles[0]
        self.coin_pos = self.coins[0]
        self.score = 0
        self.obstacle_pos = self.obstacles[0]
        self.distance_traveled = 0
        self.distance_limit = 1200

        low = np.array([0, 0])  # Minimum coordinates in each dimension
        # Maximum coordinates in each dimension
        high = np.array([len(self.hall)-1, len(self.hall)-1])
        coin_low = np.array([low, low, low,
                             low, low, low,
                             low, low, low,])
        coin_high = np.array([high, high, high,
                              high, high, high,
                              high, high, high])
        obstacle_low = np.array([low, low, low, low, low, low, low])
        obstacle_high = np.array(
            [high, [high[0], high[1] + 1], [high[0], high[1] + 2], high, high, high, high])

        # initially only one coin and obstacle eventually more
        # TODO add pixel observation space
        self.observation_space = spaces.Dict(
            {
                "barry": spaces.Box(low, high, dtype=int),
                # TODO add 3 coins or more can just be in the dictionary
                "coin1": spaces.Box(coin_low,  coin_high, shape=(9, 2), dtype=int),
                "coin2": spaces.Box(coin_low,  coin_high, shape=(9, 2), dtype=int),
                "coin3": spaces.Box(coin_low,  coin_high, shape=(9, 2), dtype=int),
                # obstacle should actually take up 3 units
                # TODO add 3 obstacles can also just be in the dictionary
                "obstacle1": spaces.Box(obstacle_low, obstacle_high, shape=(7, 2), dtype=int),
                "obstacle2": spaces.Box(obstacle_low, obstacle_high, shape=(7, 2), dtype=int),
                "obstacle3": spaces.Box(obstacle_low, obstacle_high, shape=(7, 2), dtype=int),

            }
        )
        # 0= jetpack off 1 = jetpack on
        self.action_space = spaces.Discrete(2)

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
        self.coins = [self.gen_coin()]
        self.obstacles = [self._gen_obstacle()]
        self.obstacle_pos = self.obstacles[0]
        self.coin_pos = self.coins[0]
        self.distance_traveled = 0

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
        self.distance_traveled += 1

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
                             ((coin[4][1] - 1)*self.cell_size, (coin[4][0]-1) * self.cell_size))

        # draw obstacles
        for obstacle in self.obstacles:
            i = 0
            for block in obstacle:
                i += 1
                pygame.draw.rect(self.screen, (0, 0, 0), (
                    block[1] * self.cell_size, block[0] * self.cell_size, self.cell_size, self.cell_size))
                if (i > 2):
                    break

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
        for block in self.coins[0]:
            if (np.array_equiv(self.current_barry_pos, block)):
                reward = 1
                self.coins.pop(0)
                self.coins.append(self.gen_coin())
                self._update_score()

        for block in self.obstacles[0]:
            if (np.array_equiv(self.current_barry_pos, block)):
                reward = -5
                done = True
        if (len(self.obstacles) > 1):
            for block in self.obstacles[1]:
                if (np.array_equiv(self.current_barry_pos, block)):
                    reward = -5
                    done = True
        if (len(self.obstacles) > 2):
            for block in self.obstacles[2]:
                if (np.array_equiv(self.current_barry_pos, block)):
                    reward = -5
                    done = True
        if (self.distance_traveled > self.distance_limit):
            done = True
            print("max distance traveled")
        return reward, done

    def _update_score(self):
        self.score += 1
        self.img = self.font.render(f'Score {self.score}', True, (255, 0, 0))

    def _update_coin_postion(self):
        for coin in self.coins:
            for block in coin:
                block[1] = block[1]-1
        if (self.coins[0][8][1] < 0):
            self.coins.pop(0)
            self.coins.append(self.gen_coin())
            self.coin_pos = self.coins[0]
        if (len(self.coins) < 10):
            self._gen_new_coin()

    def _gen_new_coin(self):
        newCoin = randint(1, 4)
        if (newCoin == 1):
            self.coins.append(self.gen_coin())

    def gen_coin(self):
        coin = []
        top_right = [self.np_random.integers(
            0, len(self.hall)-3), len(self.hall[1])-3]
        for i in range(0, 3):
            for j in range(0, 3):
                coin.append([top_right[0] + i, top_right[1] + j])
        return coin

    def _update_obstacle_position(self):
        for obstacle in self.obstacles:
            for block in obstacle:
                block[1] -= 1
        self.obstacle_pos = self.obstacles[0]
        if (self.obstacles[0][2][1] < 0):
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
        addCoin = False
        addObs = False
        if (len(self.coins) < 3):
            self.coins.append(self.gen_coin())
            self.coins.append(self.gen_coin())
            addCoin = True
        if (len(self.obstacles) < 3):
            self.obstacles.append(self._gen_obstacle())
            self.obstacles.append(self._gen_obstacle())
            addObs = True
        obs = {"barry": np.array(self.current_barry_pos, dtype=int),
               "coin1": np.array(self.coins[0], dtype=int),
               "coin2": np.array(self.coins[1], dtype=int),
               "coin3": np.array(self.coins[2], dtype=int),
               "obstacle1": np.array(self.obstacles[0], dtype=int),
               "obstacle2": np.array(self.obstacles[1], dtype=int),
               "obstacle3": np.array(self.obstacles[2], dtype=int),
               }
        if (addCoin):
            self.coins.pop()
            self.coins.pop()
        if (addObs):
            self.obstacles.pop()
            self.obstacles.pop()
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
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
        elif (type == 1):
            obstacle.append(
                [self.np_random.integers(1, len(self.hall)-1), len(self.hall[1])-1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]])
            obstacle.append([obstacle[0][0]+1, obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
            obstacle.append([obstacle[0][0], obstacle[0][1]])
        elif (type == 2):
            obstacle.append(
                [self.np_random.integers(0, len(self.hall)), len(self.hall[1])-2])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]+1])
            obstacle.append([obstacle[0][0]-2, obstacle[0][1]+2])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]+2])
            obstacle.append([obstacle[0][0], obstacle[0][1]+1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]])
            obstacle.append([obstacle[0][0]-2, obstacle[0][1]+1])
        elif (type == 3):
            obstacle.append(
                [self.np_random.integers(0, len(self.hall)-2), len(self.hall[1])-2])
            obstacle.append([obstacle[0][0]+1, obstacle[0][1]+1])
            obstacle.append([obstacle[0][0]+2, obstacle[0][1]+2])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]+2])
            obstacle.append([obstacle[0][0], obstacle[0][1]+1])
            obstacle.append([obstacle[0][0]-1, obstacle[0][1]])
            obstacle.append([obstacle[0][0]-2, obstacle[0][1]+1])

        return obstacle
