from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from typing import Any, List, Tuple, Optional, Callable


# ==========================================
# State Representation Component
# ==========================================
class StateHandler:
    """
    Handles conversion of MiniGrid observations to discrete (integer) states
    Mapping: (AgentX, AgentY, Direction, HoldingStatus) -> unique int
    """
    def __init__(self, env: MiniGridEnv, track_carrying: bool = False):
        self.width = env.width
        self.height = env.height
        self.track_carrying = track_carrying
        
        self.num_states: int = (self.width * self.height) * 4   # grid area x 4 directions
        
        # double num states if carrying status is tracked (for KeyEnv)
        if self.track_carrying:
            self.num_states *= 2

    def get_state_index(self, env: MiniGridEnv) -> int:
        """
        Extracts state index from env instance
        """
        x, y = env.agent_pos
        direction = env.agent_dir
        
        state_idx = (x * self.height + y) * 4 + direction # todo: verify

        # handle carrying status if tracked (update state_idx)
        if self.track_carrying and env.carrying is not None:
            state_idx += (self.width * self.height * 4)
            
        return int(state_idx)

    def get_num_states(self) -> int:
        return self.num_states

    def get_readable_state(self, state_idx: int) -> Tuple[int, int, int, int]:
        """
        Reverse mapping for debugging: index -> (x, y, dir, is_carrying)
        """
        # handle carrying status
        carrying_offset = (self.width * self.height * 4)
        is_carrying = 0
        if self.track_carrying and state_idx >= carrying_offset:
            is_carrying = 1
            state_idx -= carrying_offset
            
        # extract direction + position
        direction = state_idx % 4
        position_idx = state_idx // 4
        
        # extract x, y (the <x * height + y> part)
        y = position_idx % self.height
        x = position_idx // self.height
        
        return (x, y, direction, is_carrying)


# ==========================================
# Base class for Agent Components
# ==========================================
class BaseAgent(ABC):
    """
    Base class for all Tabular RL agents
    Handles Q-table init, action selection, hyperparameters
    """
    def __init__(self, num_states: int, num_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.n_states = num_states
        self.n_actions = num_actions
        self.lr = lr            # learning rate (alpha)
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        
        # init Q-table
        # random init # todo: consider other inits optoins
        self.q_table = np.random.uniform(low=0, high=0.01, size=(num_states, num_actions))
        
    def choose_action(self, state_idx: int, force_greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection
        :param state_idx: current state index
        :param force_greedy: controls exploration (ignores epsilon) for inference/testing
        :return : selected action index
        """
        if not force_greedy and np.random.uniform(0, 1) < self.epsilon:
            # explore:
            selected_action = np.random.randint(0, self.n_actions)
            return int(selected_action) 
        else:
            # exploit:
            values = self.q_table[state_idx]                        # get Q-table row for current state
            best_actions = np.flatnonzero(values == values.max())   # get all actions with max Q-value (1 or more)
            selected_action = np.random.choice(best_actions)        # break ties randomly (if multiple best)
            return int(selected_action) 

    @abstractmethod
    def update(self, *args) -> None:
        # todo: implement by inheriting classes
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_q_table(self, filename: str = "q_table.pkl") -> None:
        """Util for saving Q-table to file"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename: str = "q_table.pkl") -> None:
        """Util for loading Q-table from file"""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


# ==========================================
# Game-specific agents
# ==========================================
class QLearningAgent(BaseAgent):
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Q-Learning update (off-policy):
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        max_next_q = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * max_next_q
        
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (td_target - current_q)

class SARSAAgent(BaseAgent):
    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool) -> None:
        """
        SARSA update (On-Policy):
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * Q(s',a') - Q(s,a)]
        """
        next_q = self.q_table[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * next_q
        
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (td_target - current_q)

class MCAgent(BaseAgent):
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        super().__init__(n_states, n_actions, lr, gamma, epsilon)
        # MC needs to store returns for averaging
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.episode_buffer: List[Tuple[int, int, float]] = []

    def store_transition(self, state: int, action: int, reward: float) -> None:
        """Store each step for processing at end of episode"""
        step = (state, action, reward)
        self.episode_buffer.append(step)

    def update(self) -> None:
        """
        MC update:
        - executed at END of an episode
        - iterates backwards through episode buffer
        """
        G: float = 0.0 # cumulative return
        visited_pairs = set()
        
        # traverse episode backwards
        for state, action, reward in reversed(self.episode_buffer):
            G = self.gamma * G + reward
            
            # first-visit MC check
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                
                # rolling mean update: Q(s,a) = Q(s,a) + alpha * (G - Q(s,a))
                # todo: consider 1/n returns avg instead of const alpha
                old_q = self.q_table[state][action]
                self.q_table[state][action] += self.lr * (G - old_q)
        
        self.episode_buffer = [] # clear buffer


# ==========================================
# Training Manager
# ==========================================
class ExperimentRunner:
    """
    Manages training loop, logging, env interaction
    """
    def __init__(self, env: gym.Env, agent: BaseAgent, state_handler: StateHandler, max_steps: int = 100, 
                 reward_shaping_func: Optional[Callable] = None):
        self.env = env
        self.agent = agent
        self.state_handler = state_handler
        self.max_steps = max_steps
        self.reward_shaping_func = reward_shaping_func
        
        # metrics
        self.rewards_history = []
        self.steps_history = []
    
    def run_training(self, num_episodes: int, print_info: bool = False) -> Tuple[List[float], List[int]]:
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            # unwrap to MiniGridEnv for the state handler
            mg_env: MiniGridEnv = self.env.unwrapped  # error here is ok (typing)
            
            state = self.state_handler.get_state_index(mg_env)
            
            total_reward: float = 0.0
            steps = 0
            done = False
            truncated = False   # indicates if episode ended w/o reaching done state
            
            # SARSA needs the initial action
            if isinstance(self.agent, SARSAAgent):
                action = self.agent.choose_action(state)
            
            while not any([done, truncated]) and steps < self.max_steps:
                
                if not isinstance(self.agent, SARSAAgent):
                    action = self.agent.choose_action(state)
                
                # step env
                next_obs, reward, done, truncated, info = self.env.step(action)
                next_state = self.state_handler.get_state_index(mg_env)
                
                # reward shaping hook for KeyEnv ( if provided)
                if self.reward_shaping_func:
                    reward = self.reward_shaping_func(float(reward), mg_env, done, info)
                
                # update agent
                if isinstance(self.agent, QLearningAgent):
                    self.agent.update(
                        state=state, 
                        action=action, 
                        reward=float(reward), 
                        next_state=next_state, 
                        done=done
                    )
                    
                elif isinstance(self.agent, SARSAAgent):
                    next_action = self.agent.choose_action(next_state)
                    self.agent.update(
                        state=state,
                        action=action, 
                        reward=float(reward), 
                        next_state=next_state, 
                        next_action=next_action, 
                        done=done
                    )
                    action = next_action # SARSA: update current action for next loop
                    
                elif isinstance(self.agent, MCAgent):
                    self.agent.store_transition(
                        state=state, 
                        action=action, 
                        reward=float(reward)
                    )
                
                state = next_state
                total_reward += float(reward)
                steps += 1
            
            # end of episode update for MC
            if isinstance(self.agent, MCAgent):
                self.agent.update()
            
            # history update
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)
            
            # decay epsilon
            if self.agent.epsilon > 0.01:
                self.agent.epsilon *= 0.995 # todo: consider other decay rates
            
            if print_info and episode % 100 == 0:
                print(f"Episode {episode}: Steps={steps}, Reward={total_reward:.2f}, Epsilon={self.agent.epsilon:.3f}")

        return self.rewards_history, self.steps_history

    def close(self) -> None:
        self.env.close()

