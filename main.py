import gymnasium as gym
import pygame
from gymnasium.utils.play import play

from jetPackjoyRideEnv import JetPackEnv


# Register the environment
gym.register(
    id='JetPack-v0',
    entry_point='jetPackjoyRideEnv:JetPackEnv',
    kwargs={'hall': None}
)

hall = [
    ['.', '', '.', '.'],
    ['B', '.', '.', 'O'],
    ['.', '.', '.', '.'],
    ['.', '.', '.', 'C'],
]

env = gym.make('JetPack-v0', hall=hall, render_mode="human")

env.reset()


# gym.utils.play.play(env, fps=1, keys_to_action={" ": 1}, noop=0)

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
