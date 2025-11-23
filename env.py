import numpy as np
import copy
from typing import Tuple, Dict

class SnakeEnv:
    """
    Snake game environment with Gym-like interface.

    State: 9-dim feature vector
        - Direction (one-hot, 4 dims): up, right, down, left
        - Food relative position (dx, dy, 2 dims)
        - Danger detection (3 dims): front, left, right

    Actions: Relative movements
        - 0: turn left
        - 1: straight
        - 2: turn right

    Rewards:
        - +10 for eating food
        - -10 for dying
        - -0.01 per step (encourage efficiency)
    """

    # Directions: 0=up, 1=right, 2=down, 3=left
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def __init__(self, grid_size=10, seed=None):
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)

        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.steps = 0
        self.done = False

    def reset(self):
        """Reset environment to initial state."""
        # Start snake in center, moving right
        center = self.grid_size // 2
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.direction = 1  # right
        self.score = 0
        self.steps = 0
        self.done = False

        # Place food
        self._place_food()

        return self._get_state()

    def _place_food(self):
        """Place food at random empty location."""
        while True:
            x = self.rng.randint(0, self.grid_size)
            y = self.rng.randint(0, self.grid_size)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: 0=turn left, 1=straight, 2=turn right

        Returns:
            state, reward, done, info
        """
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        # Convert relative action to new direction
        # action: 0=left, 1=straight, 2=right
        if action == 0:  # turn left
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # turn right
            self.direction = (self.direction + 1) % 4
        # action == 1: straight, no change

        # Move head
        head = self.snake[0]
        dx, dy = self.DIRECTIONS[self.direction]
        new_head = (head[0] + dx, head[1] + dy)

        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True
            return self._get_state(), -10.0, True, {'reason': 'wall'}

        # Check collision with self
        if new_head in self.snake:
            self.done = True
            return self._get_state(), -10.0, True, {'reason': 'self'}

        # Move snake
        self.snake.insert(0, new_head)

        # Check if food eaten
        reward = -0.01  # small negative reward per step
        if new_head == self.food:
            reward = 10.0
            self.score += 1
            self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        self.steps += 1

        # Optional: limit steps to prevent infinite loops
        max_steps = self.grid_size * self.grid_size * 10
        if self.steps > max_steps:
            self.done = True
            return self._get_state(), -10.0, True, {'reason': 'timeout'}

        return self._get_state(), reward, self.done, {}

    def _get_state(self) -> np.ndarray:
        """
        Get current state as 9-dim feature vector.

        Returns:
            [dir_up, dir_right, dir_down, dir_left,  # one-hot direction (4)
             food_dx, food_dy,                        # relative food pos (2)
             danger_front, danger_left, danger_right] # collision ahead (3)
        """
        # Direction one-hot
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self.direction] = 1.0

        # Relative food position (normalized)
        head = self.snake[0]
        food_dx = (self.food[0] - head[0]) / self.grid_size
        food_dy = (self.food[1] - head[1]) / self.grid_size

        # Danger detection for each relative direction
        danger_front = float(self._is_collision(1))  # straight
        danger_left = float(self._is_collision(0))   # turn left
        danger_right = float(self._is_collision(2))  # turn right

        state = np.concatenate([
            direction_onehot,
            [food_dx, food_dy],
            [danger_front, danger_left, danger_right]
        ]).astype(np.float32)

        return state

    def _is_collision(self, relative_action: int) -> bool:
        """Check if relative action would cause collision."""
        # Calculate new direction
        if relative_action == 0:  # left
            new_dir = (self.direction - 1) % 4
        elif relative_action == 2:  # right
            new_dir = (self.direction + 1) % 4
        else:  # straight
            new_dir = self.direction

        # Check new position
        head = self.snake[0]
        dx, dy = self.DIRECTIONS[new_dir]
        new_pos = (head[0] + dx, head[1] + dy)

        # Wall collision
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or
            new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            return True

        # Self collision (ignore tail since it will move)
        if new_pos in self.snake[:-1]:
            return True

        return False

    def clone(self):
        """Create a deep copy for Monte Carlo rollouts."""
        return copy.deepcopy(self)

    def render(self):
        """Print ASCII representation of the game."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place food
        grid[self.food[1]][self.food[0]] = 'F'

        # Place snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 'H'  # head
            else:
                grid[y][x] = 'S'  # body

        # Print
        print('\n'.join([''.join(row) for row in grid]))
        print(f'Score: {self.score}, Steps: {self.steps}')
        print()
