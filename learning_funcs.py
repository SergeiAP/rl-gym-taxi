import gymnasium as gym
import numpy as np
from typing import Callable
from tqdm import tqdm


class TDLearning():
    
    def __init__(self, env: gym.Env, action_space: int, seed: int | None = None):
        self.env = env
        self.env_reset(seed)
        self.action_space = action_space
    
    # def _get_state_action(self,
    #                       policy: np.ndarray,
    #                       epsilon: float = 0.0) -> tuple[int, int]:
    #     if np.random.rand() > epsilon:
    #         return(self.env.env.s, policy[self.env.env.s]) 
    #     return(self.env.env.s, np.random.randint(self.action_space))
    
    def _get_epsilon_greedy_action(self, q_values: np.ndarray, epsilon: float) -> int:
        # q_values : np.ndarray["actions"]
        prob = np.ones(self.action_space) * epsilon / self.action_space
        argmax_action = np.argmax(q_values)
        prob[argmax_action] += 1 - epsilon
        action = np.random.choice(np.arange(self.action_space), p=prob)
        return action

    @classmethod
    def compute_policy_by_Q(cls, Q: np.ndarray["states", "actions"]) -> np.ndarray:
        return np.argmax(Q, axis=1)
    
    @classmethod
    def get_random_Q(cls, env: gym.Env) -> np.ndarray["states", "actions"]:
        Q = np.random.random(size=(env.observation_space.n, env.action_space.n))
        return Q
    
    def compute_episode(self,
                        Q: np.ndarray,
                        steps: int,
                        reward_func: Callable,
                        epsilon: float = 0.0,
                        seed: int | None = None) -> tuple[np.ndarray, float]:
        self.env_reset(seed)
        state = self.env.env.s
        action = reward_func.get_epsilon_greedy_action(Q[state], epsilon)
        total_reward = 0
        
        for _ in range(steps):
            state_next, reward, terminated, truncated, info = self.env.step(action)
            
            Q[state][action], action_next = reward_func(Q, state, action, reward, state_next, epsilon)
            
            state = state_next
            action = action_next
            total_reward += reward
            
            if terminated:
                return Q, total_reward
        return Q, total_reward
        
    def compute_episodes(self,
                         total_episodes: int,
                         steps: int,
                         reward_func: Callable,
                         epsilon: Callable) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        Q_history = []
        policy_history = []
        rewards = []
        Q = self.get_random_Q(self.env)
        policy = self.compute_policy_by_Q(Q)
        for n in tqdm(range(1, total_episodes+1)):
            Q, total_reward = self.compute_episode(Q, steps, reward_func, epsilon(n))
            policy = self.compute_policy_by_Q(Q)
            Q_history.append(np.copy(Q))
            policy_history.append(np.copy(policy))
            rewards.append(total_reward)
        return Q_history, policy_history, rewards
    
    def compute_experiments_episodes(self,
                                     num_experiments: int,
                                     num_experiments_pi: int,
                                     total_episodes: int,
                                     steps: int,
                                     reward_func: Callable,
                                     epsilon: Callable,
                                     seed: int | None = None
                                     ) -> tuple[list, list]:

        results = []
        Q_arr = [self.get_random_Q(self.env) for _ in range(num_experiments)]
        policy_arr = [self.compute_policy_by_Q(Q) for Q in Q_arr]
        
        for n in tqdm(range(1, total_episodes+1)):
            Q_arr = [self.compute_episode(Q_arr[i], steps, reward_func, epsilon(n))[0] for i in range(num_experiments)]
            policy_arr = [self.compute_policy_by_Q(Q) for Q in Q_arr]
            result = [self.conduct_experiments_pi(policy, num_experiments=num_experiments_pi) for policy in policy_arr]
            results.append([[x[0], x[1]] for x in result])
        self.env_close()
        return results, policy_arr
            
    def env_close(self) -> None:
        self.env.close()
        
    def env_reset(self, seed: int | None = None):
        self.env.reset(seed=seed)

    def conduct_experiments_pi(self, policy: np.ndarray, seed: int | None = None, num_experiments: int = 1000, steps: int = 1000):
        num_steps, total_reward = [], []
        for _ in range(num_experiments):
            self.env_reset(seed)
            num_steps.append(0)
            total_reward.append(0)
            for _ in range(steps):
                observation, reward, terminated, truncated, info = self.env.step(policy[self.env.env.s])
                # for eacg experiment 1 value
                total_reward[-1] += reward
                num_steps[-1] += 1
                if terminated:
                    break
        self.env_close()
        return np.mean(total_reward), np.mean(num_steps)
