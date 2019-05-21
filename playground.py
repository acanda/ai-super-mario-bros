from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
print(env.observation_space)

print(env.get_action_meanings())
print(env.get_keys_to_action())

done = True
totalReward = 0
maxReward = 0
for step in range(2000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    #print(f"State: {state.shape} {state}")
    #print(f"Info: {info}")
    totalReward += reward
    if totalReward > maxReward:
        maxReward = totalReward
    print(f"{step} ({400 - info['time']}s): {action} -> {reward} (total: {totalReward}, max: {maxReward})")
    env.render(mode='human')


env.close()
