import gymnasium as gym
import pygame
from gymnasium.utils.play import play
from gymnasium.utils.env_checker import check_env

from jetPackjoyRideEnv import JetPackEnv


# Register the environment
gym.register(
    id='JetPack-v1',
    entry_point='jetPackjoyRideEnv:JetPackEnv',
)

hall = [
    ['.', '', '.', '.'],
    ['B', '.', '.', 'O'],
    ['.', '.', '.', '.'],
    ['.', '.', '.', 'C'],
]

env = gym.make('JetPack-v1', render_mode="rgb_array")
env.reset()
# gym.utils.env_checker.check_env(env)

gym.utils.play.play(env, fps=24, keys_to_action={' ': 1}, noop=0)

"""
env.render()
for i in range(0, 10):
    pygame.event.get()
    observation, reward, done, truncated, infor = env.step(1)
    print(observation)
    env.render()
    if (done):
        break
    pygame.time.wait(200)

print(observation)
print(reward)
"""
