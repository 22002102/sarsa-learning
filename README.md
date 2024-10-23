# SARSA Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm

## SARSA LEARNING FUNCTION
```
Name: SANJAY S
Register Number: 212222230132
```
## PROGRAM :
#### SARSA Learning function:
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * Q[next_state, np.argmax(Q[next_state])] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alphas[e] * td_error

            state = next_state
            action = next_action

        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])

    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
#### optimal policy :
![image](https://github.com/user-attachments/assets/b468eca7-cbad-4527-ba30-875d2c857e6b)

#### optimal value function :
![image](https://github.com/user-attachments/assets/434257f1-4a46-47c7-b1c4-980e7175a5e1)
![image](https://github.com/user-attachments/assets/01b286dd-6dee-4d7c-a2b2-f8d5ea65ef45)

#### success rate for the optimal policy :
![image](https://github.com/user-attachments/assets/13c90dbc-5291-4990-b295-534ae66bac1d)

#### plot and state value functions of Monte Carlo method:
![image](https://github.com/user-attachments/assets/3533f9a5-b707-4031-a6f4-76a2602adf62)
![image](https://github.com/user-attachments/assets/c48a657b-5a8a-4b59-92bc-420805bfadcf)

#### plot and state value functions of SARSA learning:
![image](https://github.com/user-attachments/assets/48eebaa5-ba68-4985-893e-61f01a8a84d9)
![image](https://github.com/user-attachments/assets/d52c7651-066f-4d3c-a9b4-3c7b3f97bf98)

## RESULT:

Write your result here
