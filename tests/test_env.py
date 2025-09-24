"""pytest tests for drilling environment"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drilling_env.drilling_env import DrillingEnv
import numpy as np
import pytest


class TestDrillingEnvironment:
    """Test class for drilling environment functionality"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.env = DrillingEnv()
    
    def test_environment_creation(self):
        """Test that environment can be created successfully"""
        assert self.env is not None
        assert hasattr(self.env, 'action_space')
        assert hasattr(self.env, 'observation_space')
    
    def test_reset_functionality(self):
        """Test that reset function works correctly"""
        obs = self.env.reset()
        
        # Check that observation is returned
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        
        # Check observation shape matches observation space
        assert obs.shape == self.env.observation_space.shape
        
        # Check that all values are within expected bounds
        assert self.env.observation_space.contains(obs)
    
    def test_step_functionality(self):
        """Test that step function works correctly"""
        # Reset environment first
        initial_obs = self.env.reset()
        
        # Take a random action
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        
        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation validity
        assert obs.shape == self.env.observation_space.shape
        assert self.env.observation_space.contains(obs)
        
        # Check action was valid
        assert self.env.action_space.contains(action)
    
    def test_action_space_bounds(self):
        """Test that action space has correct bounds"""
        # Test multiple random actions
        for _ in range(10):
            action = self.env.action_space.sample()
            assert self.env.action_space.contains(action)
    
    def test_observation_space_bounds(self):
        """Test that observations stay within bounds"""
        self.env.reset()
        
        # Take several steps and check observations
        for _ in range(5):
            action = self.env.action_space.sample()
            obs, _, _, _ = self.env.step(action)
            assert self.env.observation_space.contains(obs)
    
    def test_render_functionality(self):
        """Test that render function doesn't crash"""
        self.env.reset()
        
        # Render should not raise an exception
        try:
            self.env.render()
            render_works = True
        except Exception:
            render_works = False
        
        assert render_works
    
    def test_reward_calculation(self):
        """Test that rewards are calculated properly"""
        self.env.reset()
        
        # Take several steps and check rewards
        for _ in range(5):
            action = self.env.action_space.sample()
            _, reward, _, _ = self.env.step(action)
            
            # Reward should be a finite number
            assert isinstance(reward, (int, float))
            assert np.isfinite(reward)


def test_environment_integration():
    """Integration test for complete environment workflow"""
    env = DrillingEnv()
    
    # Complete episode workflow
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(10):  # Run for 10 steps
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    # Check that we completed some steps
    assert steps > 0
    assert isinstance(total_reward, (int, float))
    assert np.isfinite(total_reward)


if __name__ == "__main__":
    pytest.main([__file__])