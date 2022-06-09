"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/06/06
Description: Implementing DDPG algorithm on the Inverted Pendulum Problem.
"""
"""
## Introduction
**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continous actions.
It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG,
which can operate over continuous action spaces.
This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)
## Problem
We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.
What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.
## Quick theory
Just like the Actor-Critic method, we have two networks:
1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or bad (negative value)
given a state and an action.
DDPG uses two more techniques not present in the original DQN:
**First, it uses two Target networks.**
**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.
Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this entire game after every move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).
**Second, it uses Experience Replay.**
We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.
Now, let's see how is it implemented.
"""
import gym
import os
import tensorflow as tf
from tensorflow.keras import Input, layers, models, backend

import numpy as np
import matplotlib.pyplot as plt

import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from other_scripts.openai_ros_common import StartOpenAI_ROS_Environment
from other_scripts.task_commons import LoadYamlFileParamsTest
from other_scripts.openai_ros_common import ROSLauncher

from model_def import build_model_v15 as model_sel

"""
We use [OpenAIGym](http://gym.openai.com/docs) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.
"""
path = os.getcwd()
parent = os.path.dirname(path)
parent = os.path.dirname(parent)
ws_path = os.path.dirname(parent)

rospy.set_param('/fssimulator/ros_ws_abspath', ws_path)

ROSLauncher(rospackage_name="reinforcement_learning",
            launch_file_name="start_training.launch",
            ros_ws_abspath=ws_path)

time.sleep(3.0)

# Load Params from the desired fssimulatorYaml file
LoadYamlFileParamsTest(rospackage_name="reinforcement_learning",
                       rel_path_from_package_to_file="config",
                       yaml_file_name="fs_simulator_openai_qlearn.yaml")

rospy.init_node('fssimulator_maze', anonymous=True, log_level=rospy.WARN)

# Init OpenAI_ROS ENV
task_and_robot_environment_name = rospy.get_param(
    '/fssimulator/task_and_robot_environment_name')
env = StartOpenAI_ROS_Environment(
    task_and_robot_environment_name)

# Create the Gym environment
rospy.loginfo("Gym environment done")
rospy.loginfo("Starting Learning")

# Set the logging system
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('reinforcement_learning')
filepath_model_load = pkg_path + "/models/v15/driverless_model_v15_1.h5"  # 'models/v2/driverless_model.h5'
outdir = pkg_path + '/training_results'
env = wrappers.Monitor(env, outdir, force=True)
rospy.loginfo("Monitor Wrapper started")

last_time_steps = np.ndarray(0)

num_states = env.observation_space.shape[0]
screen_size = np.array(env.observation_space.shape)
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound_angle = 0.9  # env.action_space.high[0]
lower_bound_angle = -0.9  # env.action_space.low[0]

upper_bound_speed = env.action_space.high[1]
lower_bound_speed = env.action_space.low[1]

print("Max Value of Angle ->  {}".format(upper_bound_angle))
print("Min Value of Angle ->  {}".format(lower_bound_angle))

print("Max Value of Speed ->  {}".format(upper_bound_speed))
print("Min Value of Speed ->  {}".format(lower_bound_speed))

"""
To implement better exploration by the Actor network, we use noisy perturbations, specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, 480, 640, 3)).astype(np.uint8)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions)).astype(np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1)).astype(np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, 480, 640, 3)).astype(np.uint8)

        self.state_batch = np.zeros((self.batch_size, 480, 640, 3)).astype(np.uint8)
        self.action_batch = np.zeros((self.batch_size, num_actions)).astype(np.float32)
        self.reward_batch = np.zeros((self.batch_size, 1)).astype(np.float32)
        self.next_state_batch = np.zeros((self.batch_size, 480, 640, 3)).astype(np.uint8)

        self.state_batch_tf = tf.Variable(self.state_batch)
        self.action_batch_tf = tf.Variable(self.action_batch)
        self.reward_batch_tf = tf.constant(self.reward_batch)
        self.next_state_batch_tf = tf.Variable(self.next_state_batch)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        # Change to float32 is required for next steps

        state_batch = tf.cast(state_batch, dtype=tf.float32)
        next_state_batch = tf.cast(next_state_batch, dtype=tf.float32)
        # reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        target_actions = target_actor(next_state_batch)
        y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tf.gradients(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tf.gradients(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        self.state_batch = self.state_buffer[batch_indices]
        self.action_batch = self.action_buffer[batch_indices]
        self.reward_batch = self.reward_buffer[batch_indices]
        self.next_state_batch = self.next_state_buffer[batch_indices]

        backend.set_value(self.state_batch_tf, self.state_batch)
        backend.set_value(self.action_batch_tf, self.action_batch)
        # backend.set_value(self.reward_batch_tf, self.reward_batch)
        backend.set_value(self.next_state_batch_tf, self.next_state_batch)

        # self.state_batch_tf = tf.constant(self.state_batch)
        # self.action_batch_tf = tf.constant(self.action_batch)
        self.reward_batch_tf = tf.constant(self.reward_batch)
        # self.next_state_batch_tf = tf.constant(self.next_state_batch)

        self.update(self.state_batch_tf, self.action_batch_tf,
                    self.reward_batch_tf, self.next_state_batch_tf)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target_old(tau):
    new_weights = []
    target_variables = target_critic.get_weights()
    variables = critic_model.get_weights()
    for i, other in enumerate(critic_model.get_weights()):
        new_weights.append(variables[i] * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.get_weights()
    variables = actor_model.get_weights()
    for i, other in enumerate(actor_model.weights):
        new_weights.append(variables[i] * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation. `BatchNormalization` is used to normalize dimensions across
samples in a mini-batch, as activations can vary a lot due to fluctuating values of input
state and action.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    """
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(512, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    
    """""

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = Input(shape=(480, 640, 3))
    x = layers.Cropping2D(cropping=((182, 107), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = tf.cast(x, dtype=tf.float32)
    x = layers.Lambda(lambda m: (m / 255.0))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    # x = layers.Conv2D(filters=42, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)

    # x = layers.SpatialDropout2D(rate=0.5)(x)
    # x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)

    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(8, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    # x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    # x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    # x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    # x = layers.Dense(10, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(x)
    # outputs[0] = outputs[0][0] * upper_bound_angle
    # outputs[1] = outputs[0][1] * upper_bound_speed

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model


def get_critic():
    # State as input
    state_input = Input(shape=(480, 640, 3))
    x = layers.Cropping2D(cropping=((182, 107), (0, 0)), input_shape=screen_size)(
        state_input)
    # x = tf.cast(x, dtype=tf.float32)
    x = layers.Lambda(lambda m: (m / 255.0))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)

    # x = layers.Conv2D(filters=42, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)

    # x = layers.SpatialDropout2D(rate=0.5)(x)
    # x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)

    state_out = layers.Flatten()(x)

    # Action as input
    action_input = layers.Input(shape=(num_actions,), dtype=tf.float32)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_input])

    out = layers.Dense(8, activation="relu")(concat)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = models.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions_np = actor_model.predict(state)
    noise = noise_object()
    # Adding noise to action
    sampled_actions_np = sampled_actions_np[0] + noise

    sampled_actions_np[0] = sampled_actions_np[0] * upper_bound_angle
    sampled_actions_np[1] = sampled_actions_np[1] * upper_bound_speed

    # We make sure action is within bounds
    legal_action = np.array([np.clip(sampled_actions_np[0], lower_bound_angle, upper_bound_angle),
                             np.clip(sampled_actions_np[1], lower_bound_speed, upper_bound_speed)])

    return np.squeeze(legal_action.astype(np.float32))


"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()  # models.load_model(filepath_model_load)
critic_model = get_critic()

target_actor = get_actor()  # models.load_model(filepath_model_load)
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 400
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(2500, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 20 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        # tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state.astype(np.float32)), 0)
        tf_prev_state = np.expand_dims(prev_state, axis=0)
        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        print 'Speed:', "%.2f" % action[1], 'Angle:', "%.2f" % action[0]

        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    buffer.learn()
    # update_target_old(tau)
    update_target(target_actor.variables, actor_model.variables, tau)
    update_target(target_critic.variables, critic_model.variables, tau)

    # for (a, b) in zip(target_actor.variables, actor_model.variables):
    #    a.assign(b * tau + a * (1 - tau))

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

"""
![Graph](https://i.imgur.com/sqEtM6M.png)
"""

"""
If training proceeds correctly, the average episodic reward will increase with time.
Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.
The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.
Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
more episodes to obtain good results.
"""

version = "1"
subversion = "0"

filepath_model_save = pkg_path + "/models/reinforcement/v" + version + "/actor_model_v" + version + "_" + subversion + ".h5"
actor_model.save(filepath_model_save)

filepath_model_save = pkg_path + "/models/reinforcement/v" + version + "/critic_model_v" + version + "_" + subversion + ".h5"
critic_model.save(filepath_model_save)

filepath_model_save = pkg_path + "/models/reinforcement/v" + version + "/target_actor_model_v" + version + "_" + subversion + ".h5"
target_actor.save(filepath_model_save)

filepath_model_save = pkg_path + "/models/reinforcement/v" + version + "/target_critic_model_v" + version + "_" + subversion + ".h5"

target_critic.save(filepath_model_save)

print('Done training')

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# Save the weights
# actor_model.save_weights("pendulum_actor.h5")
# critic_model.save_weights("pendulum_critic.h5")

# target_actor.save_weights("pendulum_target_actor.h5")
# target_critic.save_weights("pendulum_target_critic.h5")
env.close()
"""
Before Training:
![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes:
![after_img](https://i.imgur.com/eEH8Cz6.gif)
"""
