import numpy as np
from abc import ABC, abstractmethod


class RewardFunc(ABC):
    
    @abstractmethod
    def __init__(self, alpha: float, gamma: float, action_space: int):
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
    
    @abstractmethod
    def __call__(self, 
                 Q: np.ndarray, 
                 s: int,
                 a: int,
                 reward: float,
                 s_next: int,
                 epsilon: float) -> tuple[float, int]:
        pass

    def get_epsilon_greedy_action(self, q_values: np.ndarray, epsilon: float,) -> int:
        # q_values : np.ndarray["actions"]
        prob = np.ones(self.action_space) * epsilon / self.action_space
        argmax_action = np.argmax(q_values)
        prob[argmax_action] += 1 - epsilon
        action = np.random.choice(np.arange(self.action_space), p=prob)
        return action


class SARSAReward(RewardFunc):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, 
                 Q: np.ndarray, 
                 s: int,
                 a: int,
                 reward: float,
                 s_next: int,
                 epsilon: int) -> tuple[float, int]:
        a_next = self.get_epsilon_greedy_action(Q[s_next], epsilon)
        q_next_action = Q[s_next][a_next] 
        q_current = Q[s][a]
        reward = (
            q_current + 
            self.alpha * (reward + self.gamma * q_next_action - q_current))
        return reward, a_next


class SARSAMeanReward(RewardFunc):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, 
                 Q: np.ndarray, 
                 s: int,
                 a: int,
                 reward: float,
                 s_next: int,
                 epsilon: int) -> tuple[float, int]:
        a_next = self.get_epsilon_greedy_action(Q[s_next], epsilon)
        q_current = Q[s][a]
        reward = (
            q_current + 
            self.alpha * (reward + self.gamma * np.mean(Q[s_next]) - q_current))
        return reward, a_next
    
    
class QReward(RewardFunc):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self,
                 Q: np.ndarray, 
                 s: int,
                 a: int,
                 reward: float,
                 s_next: int,
                 epsilon: int) -> tuple[float, int]:
        a_next = self.get_epsilon_greedy_action(Q[s_next], epsilon)
        q_current = Q[s][a]
        reward = (
            q_current + 
            self.alpha * (reward + self.gamma * np.max(Q[s_next]) - q_current))
        return reward, a_next
