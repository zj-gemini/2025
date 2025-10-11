# --- Proximal Policy Optimization (PPO) from Scratch for CartPole ---
#
# Goal: Learn a policy to balance a pole on a cart by moving left or right.
#
# This script demonstrates the PPO algorithm, a state-of-the-art method in
# Reinforcement Learning. This version is modified to be an "Actor-Only" method,
# which is similar to the REINFORCE algorithm but uses PPO's clipped objective
# for more stable updates. It does not use a "Critic" network.
#
# 1.  The Actor (Policy Network): Decides which action to take (e.g., move left/right).
#
# --- The PPO Algorithm Steps ---
#
# 1.  Data Collection: The Actor plays out a full "episode" in the environment
#     (e.g., until the pole falls) and we store all the states, actions, and rewards.
#
# 2.  Advantage Calculation: We use the Critic's estimates and the collected rewards
#     to calculate the "advantage" for each step. Since there is no Critic, we will
#     use the discounted rewards-to-go (also called returns) as our advantage estimate.
#     This tells the policy how good the outcome was following an action.
#
# 3.  Policy Update (The PPO "Magic"): We update the Actor and Critic for several
#     epochs using the collected data. The key innovation in PPO is the "clipped
#     surrogate objective function." This special loss function prevents the policy
#     from changing too drastically in one update, which leads to more stable and
#     reliable training.

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np

# --- Hyperparameters ---
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor for future rewards
PPO_EPSILON = 0.2  # Epsilon for the clipped surrogate objective
ENTROPY_BETA = 0.01  # Weight for the entropy bonus
PPO_EPOCHS = 10  # Number of optimization epochs per data collection
TOTAL_EPISODES = 1000  # Total episodes to train for
LOG_INTERVAL = 50  # Print logs every 50 episodes

# --- Environment Setup ---
env = gym.make("CartPole-v1", disable_env_checker=True)
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n


# --- 1. Policy Network Definition (Actor-Only) ---
class Policy(nn.Module):
    """
    A policy network for the Actor. This version does not have a Critic head.
    """

    def __init__(self):
        super().__init__()
        # Actor-specific layers
        self.policy_layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
            nn.Softmax(dim=-1),  # Outputs probabilities for each action
        )

    def forward(self, state):
        # state shape: (batch_size, STATE_SIZE) or (STATE_SIZE,)
        action_probs = self.policy_layers(state)
        return action_probs


# --- 2. Data Collection ---
def collect_trajectory(model: nn.Module):
    """
    Plays one full episode in the environment to collect a trajectory.
    A trajectory consists of states, actions, rewards, and the log probabilities
    of the actions taken.
    """
    states, actions, rewards, log_probs, dones = [], [], [], [], []
    state = env.reset()[0]
    done = False

    # Let T be the length of the episode.
    while not done:
        # state shape: (STATE_SIZE,)
        state_t = torch.tensor(state, dtype=torch.float32)

        # Get action probabilities and state value from the network
        # action_probs shape: (ACTION_SIZE,)
        action_probs = model(state_t)
        dist = Categorical(action_probs)
        # action shape: (1,) (scalar tensor)
        action = dist.sample()

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action.item())

        # Store the collected data
        # states: list of T tensors, each shape (STATE_SIZE,)
        states.append(state_t)
        # actions: list of T tensors, each shape (1,)
        actions.append(action)
        # rewards: list of T floats
        rewards.append(reward)
        # log_probs: list of T tensors, each shape (1,)
        log_probs.append(dist.log_prob(action))
        # values: list of T tensors, each shape (1,)
        # dones: list of T booleans
        dones.append(done)

        state = next_state

    # Each list has length T
    return states, actions, rewards, log_probs, dones


# --- 3. Return Calculation (Rewards-To-Go) ---
def calculate_returns(rewards):
    """
    Calculates the discounted rewards-to-go for each step in a trajectory.
    This will be used as the "advantage" signal for the policy update.
    """
    returns = []
    discounted_reward = 0

    # Iterate backwards through the trajectory
    for i in reversed(range(len(rewards))):
        # The return is the current reward + the discounted return from the next step
        discounted_reward = rewards[i] + GAMMA * discounted_reward
        returns.insert(0, discounted_reward)

    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize returns for more stable training
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return torch.tensor(
        # returns shape: (T,)
        returns,
        dtype=torch.float32,
    )


# --- 4. PPO-Style Update Step ---
def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
):
    """Performs the PPO optimization step for a collected trajectory."""
    for _ in range(PPO_EPOCHS):
        # Re-evaluate the actions taken to get the current policy's log probabilities
        # states shape: (T, STATE_SIZE), new_probs shape: (T, ACTION_SIZE)
        new_probs = model(torch.stack(states))
        new_dist = Categorical(new_probs)
        # new_log_probs shape: (T,)
        new_log_probs = new_dist.log_prob(torch.stack(actions))

        # --- Calculate the PPO Loss ---
        # 1. Policy (Actor) Loss: The clipped surrogate objective
        # old_log_probs shape: (T,), ratio shape: (T,). `returns` are used as advantages.
        ratio = torch.exp(new_log_probs - torch.stack(old_log_probs).detach())
        # returns shape: (T,), surr1/surr2 shape: (T,)
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * returns
        # actor_loss shape: scalar
        actor_loss = -torch.min(surr1, surr2).mean()

        # 2. Entropy Bonus: Encourages exploration
        # entropy() shape: (T,), entropy_loss shape: scalar
        entropy_loss = -new_dist.entropy().mean()

        # Total Loss
        loss = actor_loss + ENTROPY_BETA * entropy_loss

        # --- Perform Gradient Update ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# --- 5. Main Training Loop ---
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
recent_rewards = deque(maxlen=100)  # Store rewards of the last 100 episodes

for episode in range(TOTAL_EPISODES):
    # Collect a new trajectory of data
    # All returned lists have length T (trajectory length)
    states, actions, rewards, log_probs, dones = collect_trajectory(model)

    # Calculate discounted returns (rewards-to-go)
    # returns is a tensor of shape (T,)
    returns = calculate_returns(rewards)

    # Update the policy and value networks using PPO
    ppo_update(model, optimizer, states, actions, log_probs, returns, returns)

    # Store the total reward for this episode
    total_reward = sum(rewards)
    recent_rewards.append(total_reward)

    # Logging
    if (episode + 1) % LOG_INTERVAL == 0:
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        print(
            f"Episode {episode + 1}, Last Reward: {total_reward}, Avg Reward (last 100): {avg_reward:.2f}"
        )

print("\nTraining complete!")
