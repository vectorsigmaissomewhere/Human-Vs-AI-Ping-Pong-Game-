import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN

# -----------------------------
# Custom Pong Environment
# -----------------------------
class SimplePongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super(SimplePongEnv, self).__init__()

        # Observation space: ball_x, ball_y, vel_x, vel_y, paddle_y
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
        )

        # Actions: 0 = stay, 1 = up, 2 = down
        self.action_space = spaces.Discrete(3)

        # Speeds
        self.paddle_speed = 0.03
        self.ball_speed = 0.03

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = 0.5
        self.ball_y = 0.5

        # Randomize initial direction
        self.vel_x = self.ball_speed * np.random.choice([-1, 1])
        self.vel_y = self.ball_speed * np.random.uniform(-1, 1)

        self.paddle_y = 0.5
        self.done = False

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Move paddle
        if action == 1:   # up
            self.paddle_y += self.paddle_speed
        elif action == 2: # down
            self.paddle_y -= self.paddle_speed

        self.paddle_y = np.clip(self.paddle_y, 0, 1)

        # Move ball
        self.ball_x += self.vel_x
        self.ball_y += self.vel_y

        # Bounce off top/bottom
        if self.ball_y <= 0 or self.ball_y >= 1:
            self.vel_y *= -1

        reward = 0
        terminated = False
        truncated = False

        # Paddle hit/miss
        if self.ball_x >= 1:
            if abs(self.ball_y - self.paddle_y) < 0.2:
                reward = 1
                self.vel_x *= -1
            else:
                reward = -1
                terminated = True

        # Bounce on left side
        if self.ball_x <= 0:
            self.vel_x *= -1

        # Reward shaping: encourage paddle to follow ball
        reward += 1 - abs(self.ball_y - self.paddle_y)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Normalize velocities to [-1,1] range
        norm_vel_x = np.clip(self.vel_x / self.ball_speed, -1, 1)
        norm_vel_y = np.clip(self.vel_y / self.ball_speed, -1, 1)
        return np.array([self.ball_x, self.ball_y, norm_vel_x, norm_vel_y, self.paddle_y], dtype=np.float32)


# -----------------------------
# Training
# -----------------------------
if __name__ == "__main__":
    env = SimplePongEnv()

    model = DQN(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-3, buffer_size=50000,
        exploration_fraction=0.1, exploration_final_eps=0.05,
        target_update_interval=500, train_freq=4, batch_size=32,
    )

    print("Training Pong AI ... (this may take a while)")
    model.learn(total_timesteps=500000)  # train longer for stronger AI
    model.save("pong_ai_model")
    print("Training complete! Model saved as pong_ai_model.zip")
