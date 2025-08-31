import numpy as np
from dataclasses import dataclass

# --- Q-Learning from Scratch for a Grid World ---
#
# This script demonstrates the Q-learning algorithm, a fundamental concept in
# Reinforcement Learning (RL), implemented from scratch using NumPy.
#
# --- The Problem: A Grid World ---
# The goal is for an agent to learn the optimal path from a start position (S)
# to a goal (G) on a 2D grid, while avoiding holes (H) and walls (#).
#
# - Agent (A): The entity that moves through the grid.
# - States (S): Each cell in the grid represents a state.
# - Actions (a): The agent can move up, down, left, or right.
# - Rewards (R): The agent receives feedback from the environment after each action:
#   - Goal (G): +10 reward 💰
#   - Hole (H): -10 reward 💀
#   - Any other move: -1 reward 👣 (to encourage finding the shortest path)
#
# --- The Algorithm: Q-Learning ---
# Q-learning is a model-free, off-policy RL algorithm that learns a policy,
# which tells an agent what action to take under what circumstances.
#
# - Q-table: The core of the algorithm is a table (Q_table) that stores the
#   "quality" (Q-value) of taking a certain action 'a' in a certain state 's'.
#   The dimensions are (states, actions).
#
# - The Learning Process: The agent learns by repeatedly taking actions and
#   updating the Q-table based on the rewards it receives. The update uses the
#   Bellman equation, which incorporates the immediate reward and the maximum
#   possible future reward from the next state.
#
# - Exploration vs. Exploitation: To ensure the agent discovers the entire grid
#   and doesn't just stick to the first path it finds, it uses an epsilon-greedy
#   strategy. With a probability (epsilon), which decays over time, it chooses
#   a random action (explores); otherwise, it chooses the best-known action (exploits).


# --- 1. Environment Class ---
class GridWorld:
    """
    Represents the Grid World environment.
    Encapsulates the grid layout, state transitions, and rewards, following a
    structure similar to OpenAI's Gym environments.
    """

    # Define grid characters and rewards as class attributes for clarity
    START = "S"
    GOAL = "G"
    HOLE = "H"
    WALL = "#"
    EMPTY = "."

    REWARD_GOAL = 10
    REWARD_HOLE = -10
    REWARD_STEP = -1
    REWARD_WALL = -5

    def __init__(self, grid: np.ndarray):
        """
        Initializes the GridWorld environment.

        Args:
            grid: The 2D NumPy array representing the environment layout.
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start_pos = tuple(np.argwhere(self.grid == self.START)[0])
        self.goal_pos = tuple(np.argwhere(self.grid == self.GOAL)[0])

        # The agent's current position is the state of the environment
        # Shape: (row, col) tuple
        self.current_pos = self.start_pos

        # Define actions and their corresponding movements (dy, dx)
        # 0: up, 1: right, 2: down, 3: left
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.num_actions = len(self.actions)

    def reset(self) -> tuple[int, int]:
        """Resets the agent to the start position and returns it."""
        self.current_pos = self.start_pos
        return self.current_pos

    def _is_valid_pos(self, row: int, col: int) -> bool:
        """Check if a position is within grid boundaries and not a wall."""
        is_within_bounds = (0 <= row < self.rows) and (0 <= col < self.cols)
        if not is_within_bounds:
            return False
        is_wall = self.grid[row, col] == self.WALL
        return not is_wall

    def step(self, action_idx: int) -> tuple[tuple[int, int], float, bool]:
        """
        Executes an action, updates the environment state, and returns the
        new state, reward, and terminal status.

        Args:
            action_idx: The integer index of the action to perform.
        """
        move = self.actions[action_idx]
        tentative_row, tentative_col = (
            self.current_pos[0] + move[0],
            self.current_pos[1] + move[1],
        )

        if not self._is_valid_pos(tentative_row, tentative_col):
            # Agent hits a wall or goes off-grid; penalize and stay in place.
            reward = self.REWARD_WALL
            is_terminal = False
        else:
            # The move is valid, update the agent's position.
            self.current_pos = (tentative_row, tentative_col)

            # Determine reward and terminal status based on the new position.
            if self.current_pos == self.goal_pos:
                reward = self.REWARD_GOAL
                is_terminal = True
            elif self.grid[self.current_pos] == self.HOLE:
                reward = self.REWARD_HOLE
                is_terminal = True
            else:
                reward = self.REWARD_STEP
                is_terminal = False

        return self.current_pos, reward, is_terminal


# --- 2. Agent Class ---
class QLearningAgent:
    """
    Represents the Q-learning agent.
    Manages the Q-table and the learning process (action selection and updates).
    """

    def __init__(
        self,
        state_shape: tuple[int, int],
        num_actions: int,
        learning_rate: float,
        discount_factor: float,
    ):
        """
        Initializes the Q-learning agent.

        Args:
            state_shape: The shape of the grid (rows, cols), used to initialize the Q-table.
            num_actions: The total number of possible actions the agent can take.
            learning_rate: The learning rate (alpha) for the Q-update rule.
            discount_factor: The discount factor (gamma) for future rewards.
        """
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        # Q-table stores expected future rewards for (state, action) pairs.
        # Dimensions: (rows, cols, number_of_actions)
        self.q_table = np.zeros(state_shape + (num_actions,))

    def choose_action(self, state: tuple[int, int], epsilon: float) -> tuple[int, str]:
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state: A tuple (row, col) representing the agent's current position.
            epsilon: The current probability for exploration (choosing a random action).
        """
        if np.random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action_idx = np.random.randint(self.num_actions)
            action_type = "Explore"
        else:
            # Exploit: choose the best-known action from the Q-table
            # self.q_table[state] has shape (num_actions,)
            action_idx = np.argmax(self.q_table[state])
            action_type = "Exploit"
        return action_idx, action_type

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
    ) -> tuple[float, float]:
        """
        Updates the Q-value for a given state-action pair using the Bellman equation.
        state: The state before the action, shape (row, col).
        action: The action taken, an integer index.
        reward: The reward received.
        next_state: The resulting state, shape (row, col).
        """
        # old_q_value is a scalar
        old_q_value = self.q_table[state][action]
        # Estimate the optimal future value from the next state
        # max_future_q is a scalar, the max of Q-values for all actions from next_state.
        max_future_q = np.max(self.q_table[next_state])
        # Calculate the new Q-value
        new_q_value = old_q_value + self.lr * (
            reward + self.gamma * max_future_q - old_q_value
        )
        self.q_table[state][action] = new_q_value
        return old_q_value, new_q_value


# --- 3. Main Execution ---
def simulate_policy(agent: QLearningAgent, env: GridWorld):
    """
    Simulates the learned policy from the start state and prints the path.

    Args:
        agent: The trained Q-learning agent whose policy will be simulated.
        env: The grid world environment to simulate the policy in.
    """
    print("\nSimulating the learned path from 'S':")
    env.reset()  # Ensure the environment is at the start position
    path = [env.current_pos]
    max_steps = env.rows * env.cols  # Prevent infinite loops

    for _ in range(max_steps):
        current_state = env.current_pos
        # Choose the best action (no exploration, epsilon=0)
        action_idx, _ = agent.choose_action(current_state, epsilon=0.0)

        # Take the step in the environment
        next_state, _, is_terminal = env.step(action_idx)
        path.append(next_state)

        if is_terminal:
            break

    # Format and print the final path and outcome
    path_str = " -> ".join(f"({r}, {c})" for r, c in path)
    final_state_char = env.grid[path[-1]]
    if final_state_char == env.GOAL:
        print(f"  Path to Goal: {path_str}")
    elif final_state_char == env.HOLE:
        print(f"  Path to Hole: {path_str}")
    else:
        print(f"  Path did not reach a terminal state (max steps exceeded): {path_str}")


@dataclass(frozen=True)
class Hyperparameters:
    """
    Stores all hyperparameters for a Q-learning simulation.
    Using a frozen dataclass ensures these values are not accidentally changed.
    """

    num_episodes: int
    """The total number of training episodes (games from start to finish)."""
    learning_rate: float
    """(alpha) Determines how much new information overrides old information."""
    discount_factor: float
    """(gamma) Determines the importance of future rewards. A value closer to 1 makes the agent more farsighted."""
    epsilon_start: float
    """The starting probability of choosing a random action (exploration)."""
    epsilon_end: float
    """The minimum probability of choosing a random action."""
    epsilon_decay_rate: float
    """The rate at which epsilon decays after each episode (e.g., 0.995)."""


def run_simulation(grid: np.ndarray, hyperparameters: Hyperparameters, grid_name: str):
    """
    Initializes environment and agent, runs training, and displays policy.

    Args:
        grid: The 2D NumPy array for the environment.
        hyperparameters: The dataclass object containing all hyperparameters for the run.
        grid_name: A string name for the simulation, used for printing.
    """
    print(f"\n{'='*20} Running Simulation for: {grid_name} {'='*20}")
    print("Grid Layout:")
    print(grid)

    env = GridWorld(grid)

    # --- Agent Initialization ---
    agent = QLearningAgent(
        state_shape=grid.shape,
        num_actions=env.num_actions,
        learning_rate=hyperparameters.learning_rate,
        discount_factor=hyperparameters.discount_factor,
    )

    # --- Main Training Loop ---
    print("\n--- Starting Q-learning Training ---")
    epsilon = hyperparameters.epsilon_start
    num_episodes = hyperparameters.num_episodes

    for episode in range(num_episodes):
        # current_state is a tuple (row, col)
        current_state = env.reset()
        is_terminal = False

        # Print detailed steps for the first and last episodes to see progress
        print_details = episode == 0 or episode == num_episodes - 1
        if print_details:
            print(f"\n--- Episode {episode + 1} (epsilon: {epsilon:.3f}) ---")

        while not is_terminal:
            # Agent chooses an action
            # action_idx is a scalar integer
            action_idx, action_type = agent.choose_action(current_state, epsilon)

            # Environment responds with new state, reward, and done flag
            # next_state is a tuple (row, col)
            next_state, reward, is_terminal = env.step(action_idx)

            # Agent learns from the experience
            old_q, new_q = agent.update(current_state, action_idx, reward, next_state)

            if print_details:
                print(
                    f"  State: {current_state}, Action: {action_idx} ({action_type}), Reward: {reward}, New State: {next_state}"
                )
                print(f"    Q-value updated from {old_q:.2f} to {new_q:.2f}")

            # Move to the next state for the next iteration
            current_state = next_state

        # Decay epsilon for the next episode to reduce exploration over time
        epsilon = max(
            hyperparameters.epsilon_end,
            epsilon * hyperparameters.epsilon_decay_rate,
        )

    # --- Display the Learned Policy ---
    print("\n--- Training Complete ---")
    print("Learned Policy (best action for each state):")
    # policy_grid has shape (rows, cols)
    policy_grid = np.full(grid.shape, " ", dtype=str)
    action_arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for r in range(env.rows):
        for c in range(env.cols):
            # Keep special grid characters
            if grid[r, c] in [env.START, env.GOAL, env.HOLE, env.WALL]:
                policy_grid[r, c] = grid[r, c]
            else:
                # For regular spaces, show the best learned action
                best_action_idx = np.argmax(agent.q_table[r, c, :])
                policy_grid[r, c] = action_arrows[best_action_idx]
    print(policy_grid)

    # --- Simulate and Print the Learned Path ---
    simulate_policy(agent, env)


def main():
    """Main function to run the Q-learning simulation on multiple grids."""
    # --- Hyperparameters for the simple grid ---
    # These settings are sufficient for a small, simple environment.
    hyperparams_grid_1 = Hyperparameters(
        num_episodes=2000,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_rate=0.995,  # Decays relatively quickly
    )

    # --- Grid 1: The Original Tutorial Grid ---
    grid_1 = np.array(
        [
            ["S", ".", ".", "."],
            [".", "#", "H", "."],
            [".", ".", ".", "H"],
            [".", "G", ".", "."],
        ]
    )
    run_simulation(grid_1, hyperparams_grid_1, "Original Grid")

    # --- Hyperparameters for the complex grid ---
    # A more complex environment requires more exploration to solve.
    hyperparams_grid_2 = Hyperparameters(
        num_episodes=10000,  # More episodes to allow for more learning steps
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_rate=0.9995,  # Slower decay to encourage longer exploration
    )

    # --- Grid 2: A More Complex Grid for Generalization Test ---
    grid_2 = np.array(
        [
            ["S", ".", ".", ".", "H"],
            [".", "#", ".", "#", "."],
            [".", ".", "H", ".", "."],
            ["H", "#", ".", "#", "G"],
            [".", ".", ".", ".", "."],
        ]
    )
    run_simulation(grid_2, hyperparams_grid_2, "Complex Grid")


if __name__ == "__main__":
    main()
