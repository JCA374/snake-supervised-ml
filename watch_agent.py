import pygame
import numpy as np
from env import SnakeEnv
from model import load_model
import sys
import os

class SnakeViewer:
    """Pygame viewer to watch trained agent play."""

    def __init__(self, policy_path, grid_size=10, cell_size=40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 60

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Agent - Watching Trained Policy")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)

        self.env = SnakeEnv(grid_size=grid_size)
        self.policy = load_model(policy_path)

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

        # Draw score
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
        restart_text = self.font.render('Press SPACE to restart', True, self.WHITE)
        quit_text = self.font.render('Press Q to quit', True, self.WHITE)

        self.screen.blit(game_over_text, (self.width // 2 - 100, self.height // 2 - 60))
        self.screen.blit(score_text, (self.width // 2 - 100, self.height // 2 - 20))
        self.screen.blit(restart_text, (self.width // 2 - 150, self.height // 2 + 20))
        self.screen.blit(quit_text, (self.width // 2 - 80, self.height // 2 + 60))

        pygame.display.flip()

    def watch(self, num_episodes=10, fps=10):
        """
        Watch the agent play.

        Args:
            num_episodes: number of episodes to watch
            fps: frames per second
        """
        print(f"\n=== Watching Agent Play ===")
        print(f"Episodes: {num_episodes}")
        print("Controls: SPACE=restart, Q=quit, ESC=quit\n")

        scores = []
        episode = 0

        while episode < num_episodes:
            state = self.env.reset()
            self._draw()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return scores

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return scores

                if not self.env.done:
                    # Get action from policy
                    action = self.policy.get_action(state, deterministic=False)

                    # Take action
                    state, reward, done, info = self.env.step(action)

                    self._draw()
                    self.clock.tick(fps)

                    if done:
                        scores.append(self.env.score)
                        print(f"Episode {episode+1}/{num_episodes} | "
                              f"Score: {self.env.score} | "
                              f"Steps: {self.env.steps}")

                        self._game_over_screen()

                        # Wait for restart or quit
                        waiting = True
                        while waiting:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return scores
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_SPACE:
                                        waiting = False
                                        running = False
                                        episode += 1
                                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                                        pygame.quit()
                                        return scores

        print(f"\n✓ Watched {len(scores)} episodes")
        print(f"✓ Mean score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"✓ Max score: {np.max(scores)}")

        pygame.quit()
        return scores


if __name__ == '__main__':
    policy_path = 'data/policy_imitation.pt'

    # Allow command-line arg for policy path
    if len(sys.argv) > 1:
        policy_path = sys.argv[1]

    if not os.path.exists(policy_path):
        print(f"Error: Policy not found at {policy_path}")
        print("\nAvailable policies:")
        if os.path.exists('data/policy_imitation.pt'):
            print("  - data/policy_imitation.pt (imitation learning)")
        if os.path.exists('data/policy_mc.pt'):
            print("  - data/policy_mc.pt (Monte Carlo improved)")
        sys.exit(1)

    print(f"Loading policy from: {policy_path}")

    viewer = SnakeViewer(policy_path, grid_size=10, cell_size=40)
    viewer.watch(num_episodes=10, fps=10)
