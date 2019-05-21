from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import tensorflow as tf


def state_to_tf_input(state):
    # state has shape (240, 256, 3) and will be down sampled to (9, 16)
    resized = cv2.resize(state, (15, 16), interpolation=cv2.INTER_NEAREST)  # shape (15, 16, 3)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # shape (15, 16)
    return gray[8:, :]  # crop, keep lower half -> shape (8, 16)


def show_input(input, wait=False):
    cv2.imshow("Input for Tensorflow", cv2.resize(input, (600, 320), interpolation=cv2.INTER_NEAREST))
    if wait:
        cv2.waitKey(4)


def build_model(input_shape, actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=4, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(actions, activation="softmax"))
    return model.build()


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state = env.reset()
print(f"State shape: {state.shape}")
input = state_to_tf_input(state)
print(f"Input shape: {input.shape}")
build_model(input.shape, env.action_space.n)


done = False
step = 0
while not done and step < 5000:
    step += 1
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    input = state_to_tf_input(state)
    print(f"{step}: {action} -> {reward}")
    show_input(input, True)


