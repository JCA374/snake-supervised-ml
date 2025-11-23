import pytest
import numpy as np
from env import SnakeEnv


class TestSnakeEnv:
    def test_initialization(self):
        """Test environment initializes correctly."""
        env = SnakeEnv(grid_size=10)
        state = env.reset()

        assert env.grid_size == 10
        assert len(env.snake) == 3
        assert env.score == 0
        assert env.steps == 0
        assert env.done is False
        assert env.food is not None
        assert len(state) == 9

    def test_state_shape(self):
        """Test state vector has correct shape and values."""
        env = SnakeEnv(grid_size=10)
        state = env.reset()

        assert state.shape == (9,)
        assert state.dtype == np.float32

        # Direction one-hot should sum to 1
        assert np.sum(state[:4]) == 1.0

        # Danger values should be 0 or 1
        assert all(d in [0.0, 1.0] for d in state[6:9])

    def test_movement_straight(self):
        """Test straight movement."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        initial_head = env.snake[0]
        initial_direction = env.direction

        # Move straight
        state, reward, done, info = env.step(1)

        # Direction unchanged
        assert env.direction == initial_direction

        # Head moved in correct direction
        dx, dy = env.DIRECTIONS[initial_direction]
        expected_head = (initial_head[0] + dx, initial_head[1] + dy)
        assert env.snake[0] == expected_head

    def test_movement_turn_left(self):
        """Test turning left."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        initial_direction = env.direction

        # Turn left
        env.step(0)

        expected_direction = (initial_direction - 1) % 4
        assert env.direction == expected_direction

    def test_movement_turn_right(self):
        """Test turning right."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        initial_direction = env.direction

        # Turn right
        env.step(2)

        expected_direction = (initial_direction + 1) % 4
        assert env.direction == expected_direction

    def test_wall_collision(self):
        """Test collision with wall causes game over."""
        env = SnakeEnv(grid_size=5, seed=42)
        env.reset()

        # Force snake to edge
        env.snake = [(4, 2), (3, 2), (2, 2)]
        env.direction = 1  # facing right

        # Move into wall
        state, reward, done, info = env.step(1)

        assert done is True
        assert reward == -10.0
        assert info['reason'] == 'wall'

    def test_self_collision(self):
        """Test collision with self causes game over."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        # Create situation where snake can hit itself
        # Snake in a C-shape: head at (5,5) facing right, body includes (6,5)
        env.snake = [(5, 5), (5, 4), (5, 3), (6, 3), (6, 4), (6, 5)]
        env.direction = 1  # facing right

        # Move straight right into body at (6, 5)
        state, reward, done, info = env.step(1)  # straight into body

        assert done is True
        assert reward == -10.0
        assert info['reason'] == 'self'

    def test_food_eating(self):
        """Test eating food increases score and length."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        initial_length = len(env.snake)
        initial_score = env.score

        # Place food right in front of snake
        head = env.snake[0]
        dx, dy = env.DIRECTIONS[env.direction]
        env.food = (head[0] + dx, head[1] + dy)

        # Move straight to eat food
        state, reward, done, info = env.step(1)

        assert reward == 10.0
        assert env.score == initial_score + 1
        assert len(env.snake) == initial_length + 1

    def test_step_penalty(self):
        """Test small negative reward per step."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        # Place food far away so we don't eat it
        env.food = (0, 0)

        # Take a step that doesn't eat food or die
        state, reward, done, info = env.step(1)

        assert reward == -0.01
        assert done is False

    def test_clone(self):
        """Test cloning creates independent copy."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        # Clone environment
        env_clone = env.clone()

        # Modify original
        original_head = env.snake[0]
        env.step(1)

        # Clone should be unchanged
        assert env_clone.snake[0] == original_head
        assert env_clone.steps == 0

        # They should be independent
        env_clone.step(2)
        assert env.direction != env_clone.direction

    def test_danger_detection(self):
        """Test danger detection in state vector."""
        env = SnakeEnv(grid_size=5, seed=42)
        env.reset()

        # Place snake near wall
        env.snake = [(4, 2), (3, 2), (2, 2)]
        env.direction = 1  # facing right (toward wall)

        state = env._get_state()

        # Danger front should be True (wall ahead)
        assert state[6] == 1.0  # danger_front

    def test_deterministic_with_seed(self):
        """Test environment is deterministic with seed."""
        env1 = SnakeEnv(grid_size=10, seed=42)
        env2 = SnakeEnv(grid_size=10, seed=42)

        state1 = env1.reset()
        state2 = env2.reset()

        assert np.array_equal(state1, state2)
        assert env1.food == env2.food

    def test_reset_clears_state(self):
        """Test reset properly clears game state."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        # Play some steps
        for _ in range(10):
            if not env.done:
                env.step(1)

        # Reset
        state = env.reset()

        assert env.score == 0
        assert env.steps == 0
        assert env.done is False
        assert len(env.snake) == 3

    def test_cannot_step_when_done(self):
        """Test stepping after done raises error."""
        env = SnakeEnv(grid_size=5, seed=42)
        env.reset()

        # Force collision
        env.snake = [(4, 2), (3, 2), (2, 2)]
        env.direction = 1
        env.step(1)  # hit wall

        assert env.done is True

        # Should raise error on next step
        with pytest.raises(ValueError, match="Episode is done"):
            env.step(1)

    def test_food_never_on_snake(self):
        """Test food is never placed on snake."""
        env = SnakeEnv(grid_size=10, seed=42)

        for _ in range(100):
            env.reset()
            assert env.food not in env.snake

    def test_relative_food_position(self):
        """Test food position in state is normalized."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        state = env._get_state()
        food_dx = state[4]
        food_dy = state[5]

        # Should be normalized by grid size
        assert -1.0 <= food_dx <= 1.0
        assert -1.0 <= food_dy <= 1.0
