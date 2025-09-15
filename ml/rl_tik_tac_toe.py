import math
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint


class TicTacToe:
    """
    A 4x4 Tic-Tac-Toe game environment.
    Players are 1 and -1.
    """

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Returns a list of (row, col) tuples for empty cells."""
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, move: tuple[int, int]):
        """Places a piece on the board and switches the player."""
        if self.board[move] != 0:
            raise ValueError("Invalid move")
        self.board[move] = self.current_player
        self.current_player *= -1

    def check_winner(self) -> int:
        """
        Checks for a winner.
        Returns 1 if player 1 wins, -1 if player -1 wins, 0 for a draw or ongoing game.
        """
        # Check rows and columns
        for i in range(self.board_size):
            if abs(self.board[i, :].sum()) == self.board_size:
                return self.board[i, 0]
            if abs(self.board[:, i].sum()) == self.board_size:
                return self.board[0, i]

        # Check diagonals
        if abs(np.diag(self.board).sum()) == self.board_size:
            return self.board[0, 0]
        if abs(np.diag(np.fliplr(self.board)).sum()) == self.board_size:
            return self.board[0, -1]

        # Check for draw (no empty cells left)
        if not self.get_legal_moves():
            return 0  # Draw

        return 2  # Game is ongoing

    def get_state(self) -> tuple:
        """Returns a hashable representation of the board state."""
        return tuple(self.board.flatten())

    def clone(self) -> "TicTacToe":
        """Creates a deep copy of the game state."""
        new_game = TicTacToe(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game


class MCTSNode:
    """A node in the Monte Carlo Search Tree."""

    def __init__(self, game_state: tuple, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = None  # Will be populated during expansion


class MCTS:
    """Monte Carlo Tree Search algorithm."""

    def __init__(self, exploration_constant=1.414):
        self.exploration_constant = exploration_constant

    def search(self, initial_game: TicTacToe, simulations: int) -> tuple[int, int]:
        """
        Performs MCTS search for a given number of simulations to find the best move.
        """
        root = MCTSNode(initial_game.get_state())

        for _ in range(simulations):
            node = root
            game = initial_game.clone()

            # 1. Selection: Traverse the tree to find a promising leaf node
            while node.children and not node.untried_moves:
                node = self._select_child(node)
                game.make_move(node.move)

            # 2. Expansion: If the node is not terminal, expand it
            if game.check_winner() == 2:  # Game is ongoing
                if node.untried_moves is None:
                    node.untried_moves = game.get_legal_moves()

                if node.untried_moves:
                    move = random.choice(node.untried_moves)
                    game.make_move(move)
                    node.untried_moves.remove(move)
                    child_node = MCTSNode(game.get_state(), parent=node, move=move)
                    node.children.append(child_node)
                    node = child_node

            # 3. Simulation: Play a random game from the new node
            result = self._simulate_random_playout(game)

            # 4. Backpropagation: Update stats from the leaf up to the root
            self._backpropagate(node, result)

        # After simulations, choose the move with the most visits
        best_child = sorted(root.children, key=lambda c: c.visits, reverse=True)[0]
        return best_child.move

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Selects the best child node using the UCB1 formula."""
        log_total_visits = math.log(node.visits)
        best_score = -1
        best_child = None
        for child in node.children:
            # UCB1 formula
            win_rate = child.wins / child.visits
            exploration_term = self.exploration_constant * math.sqrt(
                log_total_visits / child.visits
            )
            score = win_rate + exploration_term
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _simulate_random_playout(self, game: TicTacToe) -> int:
        """Simulates a random game from the current state."""
        while game.check_winner() == 2:  # While game is ongoing
            move = random.choice(game.get_legal_moves())
            game.make_move(move)
        return game.check_winner()

    def _backpropagate(self, node: MCTSNode, result: int):
        """Updates the wins and visits count up the tree."""
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            # The parent's player is the one who made the move to get to the current node.
            # If the result matches the parent's player, it's a win for them.
            if current_node.parent:
                # The player who made the move to `current_node` is `current_node.parent.current_player`
                # which is `-current_node.current_player`.
                # We need to check if the result is a win for the player who made the move.
                # The player at the parent node is the one who made the move to the child.
                # The game state in the parent node has `current_player` as the one to move.
                # So, if `result` matches that player, it's a win.
                # Let's simplify: the player at a node is the one *about to move*.
                # The player who *made* the move to this node is the *opposite* player.
                # If the result is a win for that opposite player, we add a win.
                # The player at `current_node.parent` is `-game.current_player` at `current_node`.
                # Let's assume player at a node is the one who just moved to get there.
                # The player at the root is -1. The player at its child is 1.
                # If player 1 wins (result=1), it's a win for nodes where player 1 moved.
                # A bit tricky. Let's simplify: a win for player 1 (result=1) is a loss for player -1.
                # We can just check if the result matches the player who is *not* about to move.
                if result == -current_node.parent.game_state[-1]:
                    current_node.wins += 1
            current_node = current_node.parent


def train_agent(episodes: int, simulations_per_move: int):
    """
    Trains an MCTS agent by self-play.
    """
    mcts_agent = MCTS()
    for episode in tqdm(range(episodes), desc="Training MCTS Agent"):
        game = TicTacToe(board_size=4)
        while game.check_winner() == 2:  # While game is ongoing
            move = mcts_agent.search(game, simulations=simulations_per_move)
            game.make_move(move)

    print("\nTraining finished.")
    return mcts_agent


def pretty_print_board(board: np.ndarray):
    """
    Prints the Tic-Tac-Toe board with 'X', 'O', and '.' for better visualization.
    """
    # Create a character representation of the board
    char_board = np.full(board.shape, ".", dtype=str)
    char_board[board == 1] = "X"  # Player 1
    char_board[board == -1] = "O"  # Player -1

    # Print with borders
    print("┌" + "───┬" * (board.shape[1] - 1) + "───┐")
    for i, row in enumerate(char_board):
        print("│ " + " │ ".join(row) + " │")
        if i < board.shape[0] - 1:
            print("├" + "───┼" * (board.shape[1] - 1) + "───┤")
    print("└" + "───┴" * (board.shape[1] - 1) + "───┘")


def play_game(agent: MCTS):
    """
    Allows a human to play against the trained MCTS agent.
    """
    game = TicTacToe(board_size=4)
    while game.check_winner() == 2:
        print("\nCurrent Board:")
        pretty_print_board(game.board)

        if game.current_player == 1:  # Human's turn
            print("\nYour turn (Player 1).")
            legal_moves = game.get_legal_moves()
            print("Available moves:", ", ".join(map(str, legal_moves)))
            while True:
                try:
                    row, col = map(int, input("Enter your move (row col): ").split())
                    if (row, col) in legal_moves:
                        game.make_move((row, col))
                        break
                    else:
                        print("Invalid move. Try again.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter row and column as two numbers.")
        else:  # Agent's turn
            print("\nAgent's turn (Player -1)...")
            move = agent.search(game, simulations=500)
            print(f"Agent chooses move: {move}")
            game.make_move(move)

    # Game over
    print("\n--- Game Over ---")
    print("Final Board:")
    pretty_print_board(game.board)
    winner = game.check_winner()
    if winner == 1:
        print("Congratulations! You won!")
    elif winner == -1:
        print("The MCTS agent won.")
    else:
        print("It's a draw!")


def main():
    # --- Training ---
    # Note: For a 4x4 board, more episodes/simulations are needed for strong play.
    # This is a demonstration; increase these values for a more robust agent.
    num_training_episodes = 100
    simulations_per_turn = 50
    trained_agent = train_agent(num_training_episodes, simulations_per_turn)

    # --- Play against the agent ---
    play_game(trained_agent)


if __name__ == "__main__":
    main()
