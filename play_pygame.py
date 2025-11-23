import pygame
import numpy as np
from env import SnakeEnv
import os

class SnakeGame:
    """Pygame interface for human play with recording."""

    def __init__(self, grid_size=10, cell_size=40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 60  # extra space for score

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake - Record Your Gameplay")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)

        self.env = SnakeEnv(grid_size=grid_size)
        self.recording = []  # stores (state, action) pairs

    def _arrow_to_relative_action(self, arrow_direction):
        """
        Convert arrow key direction to relative action.

        Args:
            arrow_direction: 0=up, 1=right, 2=down, 3=left (absolute)

        Returns:
            relative_action: 0=turn left, 1=straight, 2=turn right
        """
        current_dir = self.env.direction

        # Calculate the difference
        diff = (arrow_direction - current_dir) % 4

        if diff == 0:
            return 1  # straight
        elif diff == 1:
            return 2  # turn right
        elif diff == 3:
            return 0  # turn left
        else:  # diff == 2 (reverse, illegal)
            return 1  # ignore, go straight

    def _draw(self):
        """Draw the game state."""
        self.screen.fill(self.BLACK)

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1
                )
                pygame.draw.rect(self.screen, (30, 30, 30), rect)

        # Draw food
        food_x, food_y = self.env.food
        food_rect = pygame.Rect(
            food_x * self.cell_size,
            food_y * self.cell_size,
            self.cell_size - 1,
            self.cell_size - 1
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)

        # Draw snake
        for i, (x, y) in enumerate(self.env.snake):
            snake_rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size - 1,
                self.cell_size - 1
            )
            color = self.BLUE if i == 0 else self.GREEN
            pygame.draw.rect(self.screen, color, snake_rect)

        # Draw score and instructions
        score_text = self.font.render(
            f'Score: {self.env.score}  Steps: {self.env.steps}',
            True,
            self.WHITE
        )
        self.screen.blit(score_text, (10, self.grid_size * self.cell_size + 10))

        pygame.display.flip()

    def _game_over_screen(self):
        """Show game over screen."""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))

        game_over_text = self.font.render('GAME OVER', True, self.RED)
        score_text = self.font.render(f'Final Score: {self.env.score}', True, self.WHITE)
        restart_text = self.font.render('Press SPACE to play again', True, self.WHITE)
        quit_text = self.font.render('Press Q to quit', True, self.WHITE)

        self.screen.blit(game_over_text, (self.width // 2 - 100, self.height // 2 - 60))
        self.screen.blit(score_text, (self.width // 2 - 100, self.height // 2 - 20))
        self.screen.blit(restart_text, (self.width // 2 - 160, self.height // 2 + 20))
        self.screen.blit(quit_text, (self.width // 2 - 100, self.height // 2 + 60))

        pygame.display.flip()

    def play_episode(self):
        """Play one episode and record (state, action) pairs."""
        state = self.env.reset()
        self._draw()

        episode_recording = []
        running = True
        move_delay = 100  # milliseconds between moves
        last_move_time = pygame.time.get_ticks()
        pending_action = None

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.KEYDOWN:
                    # Map arrow keys to absolute directions
                    if event.key == pygame.K_UP:
                        pending_action = self._arrow_to_relative_action(0)
                    elif event.key == pygame.K_RIGHT:
                        pending_action = self._arrow_to_relative_action(1)
                    elif event.key == pygame.K_DOWN:
                        pending_action = self._arrow_to_relative_action(2)
                    elif event.key == pygame.K_LEFT:
                        pending_action = self._arrow_to_relative_action(3)
                    elif event.key == pygame.K_ESCAPE:
                        return None

            # Execute move at fixed interval
            if current_time - last_move_time > move_delay:
                if not self.env.done:
                    # Default to straight if no input
                    if pending_action is None:
                        pending_action = 1  # straight

                    # Record state-action pair
                    episode_recording.append((state.copy(), pending_action))

                    # Take action
                    state, reward, done, info = self.env.step(pending_action)
                    pending_action = None

                    self._draw()
                    last_move_time = current_time

                    if done:
                        # Show game over screen
                        self._game_over_screen()

                        # Wait for user input
                        waiting = True
                        while waiting:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    return episode_recording
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_SPACE:
                                        return episode_recording
                                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                                        return episode_recording

            self.clock.tick(60)

        return episode_recording

    def record_games(self, num_games=10, save_path='data/human_demos.npz'):
        """
        Record multiple games and save to disk.

        Args:
            num_games: number of games to record
            save_path: path to save recordings
        """
        all_states = []
        all_actions = []
        game_count = 0

        print(f"\n=== Snake Game Recording ===")
        print(f"Goal: Record {num_games} games")
        print("\nControls:")
        print("  Arrow keys: Move snake")
        print("  ESC: Quit early")
        print("\nPress any key to start...")

        # Wait for initial keypress
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    waiting = False

        while game_count < num_games:
            print(f"\n--- Game {game_count + 1}/{num_games} ---")

            recording = self.play_episode()

            if recording is None:  # User quit
                break

            if len(recording) > 0:
                states, actions = zip(*recording)
                all_states.extend(states)
                all_actions.extend(actions)
                game_count += 1
                print(f"Recorded {len(recording)} steps, Score: {self.env.score}")
            else:
                print("No data recorded, try again")

        if len(all_states) > 0:
            # Save to file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                states=np.array(all_states),
                actions=np.array(all_actions)
            )
            print(f"\n✓ Saved {len(all_states)} transitions to {save_path}")
        else:
            print("\nNo data to save")

        pygame.quit()


if __name__ == '__main__':
    game = SnakeGame(grid_size=10, cell_size=40)
    game.record_games(num_games=10, save_path='data/human_demos.npz')
