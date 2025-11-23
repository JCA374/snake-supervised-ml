Nice, I like that you “reviewed” the plan like a code design doc 😄

Since you didn’t pick a specific part, I’ll give you a **minimal but coherent v1 codebase** you can actually start from:

* `env.py` – Snake environment + `clone()`
* `model.py` – policy network
* `play.py` – human play + recording (logic; you can later hook up Pygame)
* `train_imitation.py` – train on your demos
* `monte_carlo.py` – MC action selection
* `self_play.py` – generate improved data using MC

You can refine/expand later.

---

## 1. `env.py` – Snake environment with clone()

This is a **non-graphical** env using a small feature vector, so it’s fast on CPU. You can add Pygame rendering later.

```python
# env.py
import random
import copy
import numpy as np
from enum import IntEnum

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

ACTION_MEANING = {
    0: "LEFT",
    1: "STRAIGHT",
    2: "RIGHT"
}

class SnakeEnv:
    def __init__(self, width=10, height=10, seed=None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        # start roughly in the middle
        self.snake = [(self.width // 2, self.height // 2)]
        self._place_food()
        self.done = False
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = self.rng.randint(0, self.width - 1)
            y = self.rng.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def _move(self, action):
        # action: 0=left, 1=straight, 2=right (relative turn)
        if action == 0:
            self.direction = Direction((self.direction - 1) % 4)
        elif action == 2:
            self.direction = Direction((self.direction + 1) % 4)
        # if action == 1: keep same direction

        head_x, head_y = self.snake[0]
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        new_head = (head_x, head_y)
        return new_head

    def _is_collision(self, position):
        x, y = position
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if position in self.snake[1:]:
            return True
        return False

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on a finished episode. Call reset().")

        self.steps += 1
        reward = 0.0

        new_head = self._move(action)

        if self._is_collision(new_head):
            self.done = True
            reward = -10.0
            return self._get_state(), reward, self.done, {}

        # move snake
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            # normal move: pop tail
            self.snake.pop()
            # small time penalty
            reward = -0.01

        # optional: simple shaping towards food
        # (commented; turn on later if you want)
        # reward += self._shaping_towards_food(new_head)

        return self._get_state(), reward, self.done, {}

    def _shaping_towards_food(self, head):
        hx, hy = head
        fx, fy = self.food
        dist = abs(hx - fx) + abs(hy - fy)
        # just a tiny signal
        return -0.001 * dist

    def _danger_in_direction(self, dir_):
        head = self.snake[0]
        if dir_ == Direction.UP:
            test_pos = (head[0], head[1] - 1)
        elif dir_ == Direction.DOWN:
            test_pos = (head[0], head[1] + 1)
        elif dir_ == Direction.LEFT:
            test_pos = (head[0] - 1, head[1])
        else:  # RIGHT
            test_pos = (head[0] + 1, head[1])
        return 1.0 if self._is_collision(test_pos) else 0.0

    def _get_state(self):
        """
        Feature vector:
        [danger_front, danger_left, danger_right,
         food_dir_x, food_dir_y,
         dir_up, dir_right, dir_down, dir_left]
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # relative food direction (normalized)
        dx = np.sign(food_x - head_x)
        dy = np.sign(food_y - head_y)

        # directions relative to current direction
        dir_front = self.direction
        dir_left = Direction((self.direction - 1) % 4)
        dir_right = Direction((self.direction + 1) % 4)

        danger_front = self._danger_in_direction(dir_front)
        danger_left = self._danger_in_direction(dir_left)
        danger_right = self._danger_in_direction(dir_right)

        dir_one_hot = [0.0, 0.0, 0.0, 0.0]
        dir_one_hot[self.direction] = 1.0

        state = np.array([
            danger_front,
            danger_left,
            danger_right,
            dx,
            dy,
            *dir_one_hot
        ], dtype=np.float32)

        return state

    def clone(self):
        # full independent copy for Monte Carlo rollouts
        return copy.deepcopy(self)

    def render_ascii(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake[1:]:
            grid[y][x] = "o"
        hx, hy = self.snake[0]
        grid[hy][hx] = "H"
        fx, fy = self.food
        grid[fy][fx] = "F"
        print("\n".join("".join(row) for row in grid))
        print(f"Score: {self.score}, Steps: {self.steps}")
```

State dimension here is **9**.

---

## 2. `model.py` – Policy network

```python
# model.py
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim=9, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)  # logits
```

---

## 3. `play.py` – Human play + recording

Here I keep it simple: **no Pygame**, just logic. You can plug in real keyboard input later.

For now, think of this as a placeholder that you extend.

```python
# play.py
import numpy as np
from env import SnakeEnv

def get_human_action():
    """
    Placeholder: replace with actual keyboard input (e.g. Pygame).
    For now, just ask in console: 0/1/2.
    """
    while True:
        val = input("Action (0=left,1=straight,2=right, q=quit): ").strip()
        if val == "q":
            return None
        if val in ("0", "1", "2"):
            return int(val)
        print("Invalid, try again.")

def record_human_games(output_path="human_demos.npz", n_episodes=5):
    env = SnakeEnv()
    states = []
    actions = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        print(f"Episode {ep+1}/{n_episodes}")
        while not done:
            env.render_ascii()
            action = get_human_action()
            if action is None:
                print("User quit.")
                np.savez_compressed(output_path, states=np.array(states), actions=np.array(actions))
                return
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            state = next_state

    np.savez_compressed(output_path, states=np.array(states), actions=np.array(actions))
    print(f"Saved {len(states)} transitions to {output_path}")

if __name__ == "__main__":
    record_human_games()
```

---

## 4. `train_imitation.py` – Train on your demos

```python
# train_imitation.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import PolicyNet

def train_imitation(
    data_path="human_demos.npz",
    epochs=10,
    batch_size=64,
    lr=1e-3
):
    data = np.load(data_path)
    states = data["states"].astype("float32")
    actions = data["actions"].astype("int64")

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(actions)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolicyNet(state_dim=states.shape[1], n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    torch.save(model.state_dict(), "policy_imitation.pt")
    print("Saved model to policy_imitation.pt")

if __name__ == "__main__":
    train_imitation()
```

---

## 5. `monte_carlo.py` – Monte Carlo action selection

```python
# monte_carlo.py
import numpy as np
import torch

def sample_action_from_logits(logits):
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    return int(np.random.choice(len(probs), p=probs))

def monte_carlo_action(env, state, policy, n_actions=3, K=5, H=20):
    """
    env: SnakeEnv
    state: np.array state for current env
    policy: PolicyNet
    """
    policy.eval()
    action_returns = []

    for a in range(n_actions):
        returns = []
        for _ in range(K):
            env_copy = env.clone()
            s, r, done, _ = env_copy.step(a)
            G = r
            steps = 0
            while not done and steps < H:
                s_tensor = torch.from_numpy(s).float().unsqueeze(0)
                with torch.no_grad():
                    logits = policy(s_tensor)
                a2 = sample_action_from_logits(logits)
                s, r, done, _ = env_copy.step(a2)
                G += r
                steps += 1
            returns.append(G)
        action_returns.append(float(np.mean(returns)))

    best_action = int(np.argmax(action_returns))
    return best_action, action_returns
```

---

## 6. `self_play.py` – Generate improved data with MC

This uses the imitation policy as rollout policy and **Monte Carlo to choose actions**, then logs `(state, best_action)` pairs.

```python
# self_play.py
import numpy as np
import torch
from env import SnakeEnv
from model import PolicyNet
from monte_carlo import monte_carlo_action

def generate_mc_data(
    model_path="policy_imitation.pt",
    output_path="mc_demos.npz",
    n_episodes=20,
    K=5,
    H=20
):
    env = SnakeEnv()
    policy = PolicyNet(state_dim=9, n_actions=3)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))

    states = []
    actions = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            a, _ = monte_carlo_action(env, state, policy, n_actions=3, K=K, H=H)
            next_state, reward, done, _ = env.step(a)
            states.append(state)
            actions.append(a)
            state = next_state
            ep_reward += reward
        print(f"Episode {ep+1}/{n_episodes}, reward {ep_reward:.2f}, score {env.score}")

    np.savez_compressed(output_path, states=np.array(states), actions=np.array(actions))
    print(f"Saved {len(states)} MC transitions to {output_path}")

if __name__ == "__main__":
    generate_mc_data()
```

You can then:

* **Combine** `human_demos.npz` + `mc_demos.npz` and retrain the policy (maybe with higher weight on MC data).
* Repeat the self-play loop a few times to iteratively improve.

---

## Suggested workflow

1. `python play.py` → record some human demos
2. `python train_imitation.py` → get `policy_imitation.pt`
3. `python self_play.py` → generate Monte Carlo data
4. Write `train_mc.py` that loads both datasets and fine-tunes the policy

---

If you tell me which part you want to *actually run first* (human recording, imitation training, or MC loop), I can tighten that file, add a small `requirements.txt`, and help you debug step 1 from “fresh folder → running code”.
