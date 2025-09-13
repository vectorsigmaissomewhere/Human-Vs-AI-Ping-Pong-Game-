from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import numpy as np
from stable_baselines3 import PPO
import torch
from stable_baselines3 import DQN
import numpy as np

# at top-level in main.py (once)
MODEL_PATH = "models/ppo_pong_final.zip"  # path from train script
VECNORM_PATH = "models/vecnormalize.pkl"  # if you used VecNormalize

# inside PongGame class in main.py, add:
def get_state_for_agent(self):
    # build state consistent with pong_env._get_obs
    # ball_x, ball_y, ball_vx, ball_vy, paddle_y (agent), opp_paddle_y
    # normalize using same scaling as env (positions 0..1, velocities scaled by ball_speed)
    # Use the same ball_speed you trained with (e.g., 0.03)
    ball_speed = 0.03  # set to match training
    bx = (self.ball.center_x) / float(self.width)
    by = (self.ball.center_y) / float(self.height)
    # velocities approximate: compute from movement, if you want exact, track previous pos
    vx = (self.ball.velocity[0] / (self.width if self.width else 1.0)) / ball_speed
    vy = (self.ball.velocity[1] / (self.height if self.height else 1.0)) / ball_speed

    paddle_y = self.player2.center_y / float(self.height)   # agent is player2 (right)
    opp_paddle_y = self.player1.center_y / float(self.height)

    state = np.array([bx, by, vx, vy, paddle_y, opp_paddle_y], dtype=np.float32)
    return state


# in PongApp.build or PongGame.__init__
try:
    model = PPO.load(MODEL_PATH)
    vec_env = None
    # If you used VecNormalize during training, you must load its stats and use it to normalize observations:
    # from stable_baselines3.common.vec_env import VecNormalize
    # vec_env = VecNormalize.load(VECNORM_PATH, env=None)  # env=None allowed for loading stats only
    # But stable-baselines3's VecNormalize works with VecEnv; for inference, you can apply obs normalization manually if needed.
    print("Model loaded.")
except Exception as e:
    print("Could not load model:", e)
    model = None

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced * 1.1
            ball.velocity = vel.x, vel.y + offset


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    popup = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the trained AI model
        self.ai_model = DQN.load("pong_ai_model")
    
    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def update(self, dt):
        self.ball.move()
        # --- Agent (model) controls player2 if model available ---

        """
        if model is not None:
            state = self.get_state_for_agent()
            # stable-baselines3 expects 2D shape for single obs: (1, n)
            action, _ = model.predict(state, deterministic=True)
            # action is a scalar (0,1,2)
            if action == 1:
                self.player2.center_y += 4
            elif action == 2:
                self.player2.center_y -= 4
            # else stay (0)
        else:
            # fallback rule-based AI
            if self.ball.center_y > self.player2.center_y:
                self.player2.center_y += 4
            elif self.ball.center_y < self.player2.center_y:
                self.player2.center_y -= 4

        # clamp paddle
        self.player2.center_y = max(self.player2.height/2, min(self.height - self.player2.height/2, self.player2.center_y))
        ######################
        """

         # --- Player 2 (AI) control ---
        obs = self._get_ai_obs()
        action, _ = self.ai_model.predict(obs, deterministic=True)

        if action == 1:   # move up
            self.player2.center_y += 5
        elif action == 2: # move down
            self.player2.center_y -= 5

        # Keep paddle inside screen
        self.player2.center_y = float(np.clip(self.player2.center_y, self.player2.height/2, self.height - self.player2.height/2))



        # bounce off paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        # bounce ball off bottom or top
        if (self.ball.y < self.y) or (self.ball.top > self.top):
            self.ball.velocity_y *= -1

        # went off to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 1  
            if self.player2.score >= 10:
                self.show_winner("Computer Wins!")
                return
            self.serve_ball(vel=(4, 0))

        if self.ball.right > self.width:
            self.player1.score += 1  
            if self.player1.score >= 10:
                self.show_winner("You Win!")
                return
            self.serve_ball(vel=(-4, 0))


    def _get_ai_obs(self):
        """Return game state in the same format used during training."""
        ball_x = self.ball.center_x / self.width
        ball_y = self.ball.center_y / self.height
        vel_x = self.ball.velocity_x / 10.0
        vel_y = self.ball.velocity_y / 10.0
        paddle_y = self.player2.center_y / self.height
        return np.array([ball_x, ball_y, vel_x, vel_y, paddle_y], dtype=np.float32)
    
    def show_winner(self, message):
        # stop the game loop
        Clock.unschedule(self.update)

        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        msg_label = Label(text=message, font_size=24, size_hint=(1, 0.7))
        restart_btn = Button(text="Restart Game", size_hint=(1, 0.3))

        layout.add_widget(msg_label)
        layout.add_widget(restart_btn)

        self.popup = Popup(title="Result",
                           content=layout,
                           auto_dismiss=False,
                           size_hint=(.6, .4))

        # Restart game when pressing restart
        def restart_game(instance):
            self.player1.score = 0
            self.player2.score = 0
            self.player1.center_y = self.center_y
            self.player2.center_y = self.center_y
            self.serve_ball()
            Clock.schedule_interval(self.update, 1.0 / 120.0)  # restart loop
            self.popup.dismiss()

        restart_btn.bind(on_release=restart_game)

        self.popup.open()

    def on_touch_move(self, touch):
        if touch.x < self.width / 3:
            self.player1.center_y = touch.y
        #if touch.x > self.width - self.width / 3:
        #    self.player2.center_y = touch.y


class PongApp(App):
    def build(self):
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 120.0)
        return game


if __name__ == '__main__':
    PongApp().run()
