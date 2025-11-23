"""
Video recording utilities for Snake gameplay.

Captures gameplay frames and saves them as videos (GIF or MP4).
"""

import numpy as np
import pygame
import os
from pathlib import Path

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Install with: pip install imageio imageio[ffmpeg]")


class VideoRecorder:
    """Records Snake gameplay as video frames."""

    def __init__(self, save_dir='training_logs', fps=10):
        """
        Initialize video recorder.

        Args:
            save_dir: directory to save videos
            fps: frames per second for video
        """
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.recording = False

        os.makedirs(save_dir, exist_ok=True)

    def start_recording(self):
        """Start a new recording session."""
        self.frames = []
        self.recording = True

    def stop_recording(self):
        """Stop recording."""
        self.recording = False

    def capture_frame(self, surface):
        """
        Capture a pygame surface as a frame.

        Args:
            surface: pygame Surface to capture
        """
        if not self.recording:
            return

        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(surface)
        # Transpose because pygame uses (width, height, 3) but we want (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(frame)

    def save_video(self, filename, format='gif'):
        """
        Save recorded frames as video.

        Args:
            filename: output filename (without extension)
            format: 'gif' or 'mp4'

        Returns:
            path to saved video or None if failed
        """
        if not self.frames:
            print("No frames to save")
            return None

        if not IMAGEIO_AVAILABLE:
            print("Cannot save video: imageio not installed")
            return None

        # Ensure directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Create output path
        ext = 'gif' if format == 'gif' else 'mp4'
        output_path = os.path.join(self.save_dir, f"{filename}.{ext}")

        try:
            if format == 'gif':
                # Save as GIF
                imageio.mimsave(output_path, self.frames, fps=self.fps, loop=0)
            else:
                # Save as MP4
                writer = imageio.get_writer(output_path, fps=self.fps, codec='libx264')
                for frame in self.frames:
                    writer.append_data(frame)
                writer.close()

            print(f"✓ Video saved to {output_path} ({len(self.frames)} frames)")
            return output_path

        except Exception as e:
            print(f"Error saving video: {e}")
            return None

    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames = []


def record_episode(env, policy, save_path, cell_size=30, fps=10, max_steps=1000):
    """
    Record a single episode as a video.

    Args:
        env: SnakeEnv instance
        policy: PolicyNet model
        save_path: path to save video (without extension)
        cell_size: size of each grid cell in pixels
        fps: frames per second
        max_steps: maximum steps to record

    Returns:
        dict with episode stats (score, steps, video_path)
    """
    # Initialize pygame
    pygame.init()

    grid_size = env.grid_size
    window_size = grid_size * cell_size
    screen = pygame.Surface((window_size, window_size))

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 100, 255)

    # Initialize recorder
    recorder = VideoRecorder(save_dir=os.path.dirname(save_path), fps=fps)
    recorder.start_recording()

    # Reset environment
    state = env.reset()
    steps = 0

    # Run episode
    while not env.done and steps < max_steps:
        # Draw current state
        screen.fill(BLACK)

        # Draw grid
        for x in range(grid_size):
            for y in range(grid_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, WHITE, rect, 1)

        # Draw snake
        for i, (x, y) in enumerate(env.snake):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            color = GREEN if i == 0 else BLUE  # Head is green, body is blue
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 1)

        # Draw food
        fx, fy = env.food
        food_rect = pygame.Rect(fx * cell_size, fy * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, RED, food_rect)

        # Capture frame
        recorder.capture_frame(screen)

        # Get action and step
        action = policy.get_action(state, deterministic=False)
        state, reward, done, info = env.step(action)
        steps += 1

    # Draw final frame
    screen.fill(BLACK)
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)
    for i, (x, y) in enumerate(env.snake):
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        color = GREEN if i == 0 else BLUE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, WHITE, rect, 1)
    recorder.capture_frame(screen)

    # Save video
    recorder.stop_recording()
    filename = os.path.basename(save_path)
    video_path = recorder.save_video(filename, format='gif')

    pygame.quit()

    return {
        'score': env.score,
        'steps': steps,
        'video_path': video_path
    }


def record_best_episode(policy, num_episodes=10, save_path='training_logs/best_episode',
                        grid_size=10, cell_size=30, fps=10, max_steps=1000):
    """
    Record the best episode out of multiple attempts.

    Args:
        policy: PolicyNet model
        num_episodes: number of episodes to try
        save_path: path to save best video (without extension)
        grid_size: size of game grid
        cell_size: size of each cell in pixels
        fps: frames per second
        max_steps: max steps per episode

    Returns:
        dict with best episode stats
    """
    from env import SnakeEnv

    # Find best episode
    best_score = -1
    best_env = None

    for _ in range(num_episodes):
        env = SnakeEnv(grid_size=grid_size)
        state = env.reset()
        steps = 0

        while not env.done and steps < max_steps:
            action = policy.get_action(state, deterministic=False)
            state, reward, done, info = env.step(action)
            steps += 1

        if env.score > best_score:
            best_score = env.score
            best_env = env

    # Record the best episode (replay)
    if best_env is None:
        print("No valid episodes found")
        return None

    # Re-run the best episode to record it
    print(f"Recording best episode (score: {best_score})...")
    result = record_episode(best_env.__class__(grid_size=grid_size), policy,
                           save_path, cell_size, fps, max_steps)

    return result


if __name__ == '__main__':
    # Test video recording
    from env import SnakeEnv
    from model import load_model
    import os

    if os.path.exists('data/policy_imitation.pt'):
        print("Loading policy...")
        policy = load_model('data/policy_imitation.pt')

        print("Recording best episode...")
        result = record_best_episode(
            policy,
            num_episodes=5,
            save_path='training_logs/test_episode',
            grid_size=10
        )

        if result:
            print(f"\n✓ Recorded episode:")
            print(f"  Score: {result['score']}")
            print(f"  Steps: {result['steps']}")
            print(f"  Video: {result['video_path']}")
    else:
        print("No policy found. Train a model first.")
