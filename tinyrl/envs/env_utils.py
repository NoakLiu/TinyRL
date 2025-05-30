"""
Environment utilities for TinyRL
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Optional, Union


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create a gymnasium environment with optional wrappers"""
    env = gym.make(env_name, **kwargs)
    return env


class VectorizedEnv:
    """Simple vectorized environment wrapper"""
    
    def __init__(self, env_fns: List[callable]):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get environment specs from first env
        sample_env = self.envs[0]
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        
    def reset(self):
        """Reset all environments"""
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
            
        return np.array(observations), infos
    
    def step(self, actions):
        """Step all environments"""
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
            
        return (
            np.array(observations),
            np.array(rewards),
            np.array(dones),
            np.array(truncateds),
            infos
        )
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


def create_env_config(env_name: str) -> Dict[str, Any]:
    """Create environment configuration based on environment name"""
    
    # CartPole configuration
    if "CartPole" in env_name:
        return {
            "env_name": env_name,
            "state_dim": 4,
            "action_dim": 2,
            "continuous_actions": False,
            "max_episode_steps": 500,
        }
    
    # Pendulum configuration
    elif "Pendulum" in env_name:
        return {
            "env_name": env_name,
            "state_dim": 3,
            "action_dim": 1,
            "continuous_actions": True,
            "max_episode_steps": 200,
        }
    
    # LunarLander configuration
    elif "LunarLander" in env_name:
        continuous = "Continuous" in env_name
        return {
            "env_name": env_name,
            "state_dim": 8,
            "action_dim": 2 if continuous else 4,
            "continuous_actions": continuous,
            "max_episode_steps": 1000,
        }
    
    # Default configuration
    else:
        # Try to infer from environment
        try:
            env = gym.make(env_name)
            obs_space = env.observation_space
            action_space = env.action_space
            
            if hasattr(obs_space, 'shape'):
                state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
            else:
                state_dim = obs_space.n
            
            if hasattr(action_space, 'shape'):
                action_dim = action_space.shape[0] if len(action_space.shape) == 1 else np.prod(action_space.shape)
                continuous_actions = True
            else:
                action_dim = action_space.n
                continuous_actions = False
            
            env.close()
            
            return {
                "env_name": env_name,
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "continuous_actions": continuous_actions,
                "max_episode_steps": 1000,
            }
            
        except Exception as e:
            print(f"Warning: Could not infer environment config for {env_name}: {e}")
            return {
                "env_name": env_name,
                "state_dim": 4,
                "action_dim": 2,
                "continuous_actions": False,
                "max_episode_steps": 1000,
            } 