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


class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced * 10
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

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def update(self, dt):
        self.ball.move()

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
                self.show_winner("Player 2 Wins!")
                return
            self.serve_ball(vel=(4, 0))

        if self.ball.right > self.width:
            self.player1.score += 1  
            if self.player1.score >= 10:
                self.show_winner("Player 1 Wins!")
                return
            self.serve_ball(vel=(-4, 0))

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
        if touch.x > self.width - self.width / 3:
            self.player2.center_y = touch.y


class PongApp(App):
    def build(self):
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 120.0)
        return game


if __name__ == '__main__':
    PongApp().run()
