import gym_super_mario_bros

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent, element_wise_squared_loss
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils import common

from wrapper import ShrinkWrapper, DiscreteActionWrapper

import matplotlib.pyplot as plt
import time
import os

start = time.time()

tf.compat.v1.enable_v2_behavior()

env_name = 'SuperMarioBros-1-1-v2'
conv_layer_params = [
    (32, (3, 3), (1, 1)),
    (64, (3, 3), (1, 1)),
    (64, (3, 3), (1, 1))
]
fc_layer_params = [128]
learning_rate = 1e-3
replay_buffer_capacity = 100000
initial_collect_steps = 1000
batch_size = 64
num_eval_episodes = 2
max_episode_steps_train = None
max_episode_steps_eval = 1000
num_iterations = 50000
collect_steps_per_iteration = 1
log_interval = 1000
eval_interval = 5000

# create training and evaluation environments
train_env = TFPyEnvironment(suite_gym.load(env_name, max_episode_steps=max_episode_steps_train,
                                           gym_env_wrappers=[ShrinkWrapper, DiscreteActionWrapper]))
eval_env = TFPyEnvironment(suite_gym.load(env_name, max_episode_steps=max_episode_steps_eval,
                                          gym_env_wrappers=[ShrinkWrapper, DiscreteActionWrapper]))

# create DQN (deep Q-Learning network)
q_net = QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

# create deep reinforcement learning agent
tf_agent = DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=element_wise_squared_loss,
    train_step_counter=train_step_counter)
tf_agent.initialize()

# create evaluation and data collection policies
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

# create replay buffer
print("Creating replay buffer")
replay_buffer = TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


# execute the random policy in the environment for a few steps
# and record the data (observations, actions, rewards etc) in the replay buffer
print("Collecting initial random steps")
random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)

dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

# train the agent
print("Training the agent")
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)


def compute_avg_reward(environment, policy, num_episodes=10):
    total_reward = 0.0
    for episode in range(num_episodes):
        print(f"Computing reward: game {episode + 1}/{num_episodes}")

        time_step = environment.reset()
        episode_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
        total_reward += episode_reward

    avg_return = total_reward / num_episodes
    return avg_return.numpy()[0]


# Evaluate the agent's policy once before training.
avg_reward = compute_avg_reward(eval_env, tf_agent.policy, num_eval_episodes)
rewards = [avg_reward]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for coll_step in range(collect_steps_per_iteration):
        collect_step(train_env, tf_agent.collect_policy)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_reward = compute_avg_reward(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average reward = {1}'.format(step, avg_reward))
        rewards.append(avg_reward)

print(f'Finished in {(time.time() - start):.0f} s')

now = time.strftime('%Y%m%d%H%M%S', time.localtime())
filename_postfix = f'c{list(map(lambda x: x[0], conv_layer_params))}-fc{fc_layer_params}-i{num_iterations}-b{batch_size}-lr{learning_rate}'

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, rewards)
plt.ylabel('Average Reward')
plt.xlabel('Step')

os.makedirs('results', exist_ok=True)
plt.savefig(os.path.join('results', f'{now}-rewards-{filename_postfix}.png'))
plt.show()
