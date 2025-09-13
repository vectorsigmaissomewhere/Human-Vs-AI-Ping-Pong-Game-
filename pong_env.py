# pong_env.py
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

class PongEnv(gym.Env):
    """
    Simple headless Pong environment.
    State vector: [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opp_paddle_y]
    Coordinates normalized to [0,1] for positions, velocities scaled.
    Action space (Discrete(3)): 0=stay, 1=up, 2=down
    """
    metadata = {"render.modes": []}

    def __init__(self, max_steps=1000, paddle_height=0.2, paddle_speed=0.04, ball_speed=0.03):
        super().__init__()
        self.width = 1.0
        self.height = 1.0

        self.paddle_height = paddle_height
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed

        # action: stay / up / down
        self.action_space = spaces.Discrete(3)

        # observation: ball_x, ball_y, ball_vx, ball_vy, paddle_y (agent), opp_paddle_y (opponent)
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = max_steps
        self.step_count = 0

        # internal state
        self.reset()

    def reset(self, seed=None, options=None):
        self.step_count = 0

        # paddles centered on y
        self.paddle_y = 0.5
        self.opp_paddle_y = 0.5

        # ball in center, random direction
        self.ball_x = 0.5
        self.ball_y = 0.5
        angle = np.random.uniform(-math.pi/4, math.pi/4)  # moderate angle
        direction = np.random.choice([-1, 1])
        vx = direction * abs(math.cos(angle)) * self.ball_speed
        vy = math.sin(angle) * self.ball_speed
        self.ball_vx = vx
        self.ball_vy = vy

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([self.ball_x, self.ball_y, self.ball_vx / self.ball_speed,
                         self.ball_vy / self.ball_speed, self.paddle_y, self.opp_paddle_y],
                        dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0

        # apply action to agent paddle
        if action == 1:
            self.paddle_y += self.paddle_speed
        elif action == 2:
            self.paddle_y -= self.paddle_speed
        self.paddle_y = float(np.clip(self.paddle_y, 0.0, 1.0))

        # simple opponent policy: follow the ball with some noise
        if self.ball_y > self.opp_paddle_y + 0.01:
            self.opp_paddle_y += self.paddle_speed * 0.9
        elif self.ball_y < self.opp_paddle_y - 0.01:
            self.opp_paddle_y -= self.paddle_speed * 0.9
        # clamp opponent paddle
        self.opp_paddle_y = float(np.clip(self.opp_paddle_y, 0.0, 1.0))

        # update ball position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce off top/bottom
        if self.ball_y <= 0.0:
            self.ball_y = 0.0
            self.ball_vy *= -1
        if self.ball_y >= 1.0:
            self.ball_y = 1.0
            self.ball_vy *= -1

        # check collision with left paddle (opponent)
        left_paddle_x = 0.0  # x position (not used)
        if self.ball_x <= 0.05:
            # check y overlap with opp paddle
            if abs(self.ball_y - self.opp_paddle_y) <= self.paddle_height / 2:
                # bounce
                self.ball_x = 0.05
                self.ball_vx *= -1
                # add some y velocity depending on where it hit
                offset = (self.ball_y - self.opp_paddle_y) / (self.paddle_height / 2)
                self.ball_vy += offset * 0.01
            else:
                # agent scores
                reward = 1.0
                done = True

        # check collision with right paddle (agent)
        if self.ball_x >= 0.95:
            if abs(self.ball_y - self.paddle_y) <= self.paddle_height / 2:
                self.ball_x = 0.95
                self.ball_vx *= -1
                offset = (self.ball_y - self.paddle_y) / (self.paddle_height / 2)
                self.ball_vy += offset * 0.01
            else:
                # opponent scores -> negative reward
                reward = -1.0
                done = True

        # small step penalty to encourage longer rallies? (optional)
        # reward += -0.0001

        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, float(reward), done, False, {}

    def render(self, mode="human"):
        # optional: not implemented, training is headless
        pass

    def close(self):
        pass
