from gym.core import ObservationWrapper, ActionWrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.spaces import Box, Discrete
import numpy as np
import cv2


class ShrinkWrapper(ObservationWrapper):
    r"""Shrinks the observation from (240, 256, 3) to (8, 15) by resizing to (16, 15),
    converting to gray scale and finally removing the top half.
    """

    def __init__(self, env):
        super(ShrinkWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(16, 30), dtype=np.uint8)

    def observation(self, observation):
        rgb = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        cv2.imshow("Original Observation", cv2.resize(rgb, (640, 600), interpolation=cv2.INTER_NEAREST))
        resized = cv2.resize(observation, (30, 32), interpolation=cv2.INTER_NEAREST)  # shape (32, 30, 3)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # shape (32, 30)
        cropped = gray[16:, :]  # crop, keep lower half -> shape (16, 30, 3)
        cv2.imshow("Cropped Observation", cv2.resize(cropped, (600, 320), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)
        return cropped


class DiscreteActionWrapper(ActionWrapper):
    SIMPLE_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
    ]

    # a mapping of buttons to binary values
    _button_map = {
        'right': 0b10000000,
        'left': 0b01000000,
        'down': 0b00100000,
        'up': 0b00010000,
        'start': 0b00001000,
        'select': 0b00000100,
        'B': 0b00000010,
        'A': 0b00000001,
        'NOOP': 0b00000000,
    }

    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        actions = SIMPLE_MOVEMENT
        # This should be a discrete space but TF wants a space of type float
        self.action_space = Discrete(len(actions))
        self._action_map = {}
        self._action_meanings = {}
        for action, button_list in enumerate(actions):
            byte_action = 0
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

    def action(self, action):
        if isinstance(action, int):
            as_int = action
        elif isinstance(action, (np.generic, np.ndarray)) and (
                action.dtype.kind in np.typecodes['AllInteger'] and action.shape == ()):
            as_int = int(action)
        return self._action_map[as_int]

    def get_action_meanings(self):
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]
