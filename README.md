## Project Image 
![Project Screenshot](https://github.com/vectorsigmaissomewhere/Human-Vs-AI-Ping-Pong-Game-/blob/main/project_images/capture557.PNG)

## Main concepts of RL 
```text 
- States 
- Actions
- Reward functions 
= Environment dynamic(Markov Decision Processes)
- Policies 
- Tranjectories and return 
- Value function & Q-function 
```

RL Framework 
```
- Agent solves problem in a certain environment 
- at every timestep t, the agent need to choose a action a. 
- With this action the agent receives reward r
- we get the new observation of its state s
- the new state can be determined both by the action of the agent 
and also by the environment the agent is operating in 
```

## Concept of RL in Ping Pong

States
```text
States:N these are usally vector, matrix, other tensor
state provide information about which action to take in timestep t

In Ping Pong: 
vector means array of number 
and these numbers can be position of the player's paddle, 
the position of the ball and the angular velocities of the ball 
then our agent moves according the vector 

we also need in which direction the ball is moving

also take frames/pixels of the game to represent the state
now the state is a matrix/ tensor 

now agent need to learn how to play the game 
agent need to learn the circular pixel is a ball 
agent need to learn the location of the ball 


one static image is not enought to know where the ball is moving
so it need to know a stack of the last few frames 
```

Actions
```text 
in pingpong action is where the paddle moves up and down 

make a distinction between discrete action spaces and continuous action space

in discrete action space our actions are discrete, means it can 
take only a certain set of values 
in pinngpong it is move up , move down and no move 

vector for up [1.0, 0.0, 0.0]
vector for down [0.0, 1.0, 0.0]
vector for no move [0.0, 0.0, 1.0]

each column of these vector indicate which action to take 
Note: values of vector are continuous which agent takes but 
for now just look the dimension of the vector where the number is 1

In continuous action , our actions are continuous, means they can 
take on any value. 
now the action is not just moving up and down but also knowing 
how much should we move that is positive value for moving up 
0 for moving nowhere and -1 for moving down 
```

Reward function
```text 
how function an agent got for taking the action
rewardfunction(state(St), action(At), timestep t)
    return reward(Rt)
 
Reward Rt is a scalar(number) that indicates how well the agent did 

Reward of agent in ping pong: 
    if the agent send ball to nexxt player and agent scores; reward of 1 
    0 when nobody scores 
    -1 when agent misses and the next player score  

Reward function signature: 
   S * A -> R
   here S means set of states, A means set of actions, R means real numbers 
```

Environment dynamics 
```text
Environment dynmaics gives the answer like 
a state and an action is given and what will be the next state
```

Markov Decision Process 
```text 
MDP is a discrete-time stochastic control process that the
Markov property. 

According to Markov property, next state is defined 
by current state and not on other states. 
```

Policies
```text 
Behavior of our agent 

Distinguish between deterministic policies and stochastic policies 
```

Neural Networks
```text 
Neural Network is something similar to human brain. 
This have neurons and synapses. 


input -> state 
output -> action of our agent 
```

Trajectories and returns 
```text 
certain sequence of states and actions 
```

Value function and Q-function
```text 


```

## In Simplest Term for Ping Pong 
```text
1. States

Where it’s used: Input to your RL agent (neural network).

Ping pong example:

Paddle y-position

Ball position (x, y)

Ball velocity (vx, vy)

Opponent paddle y-position (optional)

If using vision → last N game frames as an image stack.

👉 The state is the agent’s “eyes” — what it knows about the game at timestep t.

2. Actions

Where it’s used: Output of the RL agent’s policy.

Ping pong example:

Discrete: {Move Up, Move Down, Stay Still}

Continuous: A real number ∈ [-1,1] = paddle velocity (negative = down, positive = up).

👉 The action is the “decision” your agent makes based on the state.

3. Reward function

Where it’s used: Signal to train the agent (optimize behavior).

Ping pong example:

+1 → agent scores

-1 → opponent scores

+0.1 → successful paddle-ball hit (shaping)

0 → otherwise

👉 The reward is the “teacher” that tells the agent how good its last action was.

4. Environment dynamics (Transition function)

Where it’s used: Defines how the world updates after an action.

Ping pong example:

Ball moves according to physics equations.

Paddle position changes when agent moves.

If ball hits wall, bounce.

If ball crosses left/right, point is scored.

👉 The dynamics are the “rules of ping pong” built into your game engine.

5. Markov Decision Process (MDP)

Where it’s used: The formal mathematical framework for the game.

Ping pong example:

States = game information

Actions = paddle movements

Rewards = scoring system

Transition = physics update

Policy = agent’s brain

👉 Your entire ping pong RL problem is an MDP: (S, A, R, T, π).

6. Policies (π)

Where it’s used: Defines how the agent selects actions.

Ping pong example:

Deterministic policy: always move paddle exactly towards predicted ball position.

Stochastic policy: with probability 0.8 move up, 0.2 move stay still (adds randomness, helps in exploration).

👉 The policy is the “brain” mapping from state → action.

7. Neural Networks

Where it’s used: Function approximator for the policy and/or value function.

Ping pong example:

Input layer: state vector (ball_x, ball_y, vx, vy, paddle_y, …)

Hidden layers: MLP / CNN if using frames.

Output layer: probabilities of actions (discrete) or continuous velocity.

👉 Neural nets are the “machinery” that learns to play ping pong.

8. Trajectories and returns

Where it’s used: When computing learning updates.

Ping pong example:

Trajectory: s0, a0, r0, s1, a1, r1, ... until the rally ends.

Return: sum of discounted rewards = total future points from current timestep.

👉 Trajectories are the “stories” of gameplay; returns measure how good they were.

9. Value function & Q-function

Where it’s used: Critic inside actor-critic methods (PPO, A2C, SAC) or in Q-learning.

Ping pong example:

Value function V(s): expected total return from state s (e.g., “if ball is near my paddle and moving slowly, I’m in a good state”).

Q-function Q(s,a): expected return if I take action a in state s (“if I move up now, I’ll likely hit the ball → good”).

👉 Value functions are the “judges” estimating how good states or actions are.
```


Resources that I have used 
```text 
About Reinforcement Learning: 
https://medium.com/@cedric.vandelaer/reinforcement-learning-an-introduction-part-1-3-866695deb4d1
Building the UI and Game function: 
https://kivy.org/doc/stable/
Gymnasium(custom environments):
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
Stable-Baseline3 DQN example: 
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
RL Baseline3 Zoo Repo: https://github.com/DLR-RM/rl-baselines3-zoo
Stable Baseline3(Github): https://github.com/DLR-RM/stable-baselines3
Learning to play Pong with PyTorch & Tinashou(Medium): https://medium.com/%40joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd
Playing Pong using Reinforcement Learning(MathWorks blog): https://blogs.mathworks.com/deep-learning/2021/03/12/playing-pong-using-reinforcement-learning/
Beating Pong using Reinforcement Learing: https://medium.com/analytics-vidhya/beating-pong-using-reinforcement-learning-part-1-dddqn-f7fbf5ad7768
```