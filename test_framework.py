#!/usr/bin/env python3
"""
Simple test script to verify TinyRL framework functionality
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_attention_models():
    """Test attention models"""
    print("Testing attention models...")
    
    try:
        from tinyrl.models.attention import FlashAttentionModel, LinearAttentionModel, HybridAttentionModel
        
        config = {
            "input_dim": 64,
            "output_dim": 128,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 2,
            "dropout": 0.1,
            "device": "cpu"
        }
        
        # Test Flash Attention
        flash_model = FlashAttentionModel(config)
        x = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, input_dim=64
        output = flash_model(x)
        assert output.shape == (2, 10, 128), f"Expected (2, 10, 128), got {output.shape}"
        print("✓ Flash Attention model works")
        
        # Test Linear Attention
        linear_model = LinearAttentionModel(config)
        output = linear_model(x)
        assert output.shape == (2, 10, 128), f"Expected (2, 10, 128), got {output.shape}"
        print("✓ Linear Attention model works")
        
        # Test Hybrid Attention
        hybrid_model = HybridAttentionModel(config)
        output = hybrid_model(x)
        assert output.shape == (2, 10, 128), f"Expected (2, 10, 128), got {output.shape}"
        print("✓ Hybrid Attention model works")
        
    except Exception as e:
        print(f"✗ Attention models test failed: {e}")
        return False
    
    return True

def test_networks():
    """Test network architectures"""
    print("\nTesting network architectures...")
    
    try:
        from tinyrl.models.networks import PolicyNetwork, ValueNetwork, QNetwork
        
        config = {
            "state_dim": 4,
            "action_dim": 2,
            "hidden_dim": 128,
            "attention_type": "hybrid",
            "use_attention": True,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "continuous_actions": False,
            "device": "cpu"
        }
        
        # Test Policy Network
        policy_net = PolicyNetwork(config)
        state = torch.randn(3, 4)  # batch_size=3, state_dim=4
        logits = policy_net(state)
        assert logits.shape == (3, 2), f"Expected (3, 2), got {logits.shape}"
        
        # Test action distribution
        action_dist = policy_net.get_action_distribution(state)
        action = action_dist.sample()
        assert action.shape == (3,), f"Expected (3,), got {action.shape}"
        print("✓ Policy Network works")
        
        # Test Value Network
        value_net = ValueNetwork(config)
        values = value_net(state)
        assert values.shape == (3,), f"Expected (3,), got {values.shape}"
        print("✓ Value Network works")
        
        # Test Q Network
        q_net = QNetwork(config)
        q_values = q_net(state)
        assert q_values.shape == (3, 2), f"Expected (3, 2), got {q_values.shape}"
        print("✓ Q Network works")
        
    except Exception as e:
        print(f"✗ Networks test failed: {e}")
        return False
    
    return True

def test_replay_buffer():
    """Test replay buffers"""
    print("\nTesting replay buffers...")
    
    try:
        from tinyrl.utils.replay_buffer import ReplayBuffer, PPOBuffer
        
        device = torch.device("cpu")
        
        # Test standard replay buffer
        buffer = ReplayBuffer(size=1000, state_dim=4, action_dim=2, device=device)
        
        # Store some transitions
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i == 9
            
            buffer.store(state, action, reward, next_state, done)
        
        # Sample batch
        batch = buffer.sample(5)
        assert batch['states'].shape == (5, 4)
        assert batch['actions'].shape == (5, 2)
        print("✓ ReplayBuffer works")
        
        # Test PPO buffer
        ppo_buffer = PPOBuffer(size=100, state_dim=4, action_dim=2, device=device)
        
        for i in range(10):
            state = np.random.randn(4)
            action = torch.randn(2)
            reward = np.random.randn()
            value = torch.randn(1)
            log_prob = torch.randn(1)
            done = i == 9
            
            ppo_buffer.store(state, action, reward, value, log_prob, done)
        
        ppo_buffer.compute_advantages_and_returns()
        data = ppo_buffer.get_all()
        assert data['states'].shape == (10, 4)
        assert data['advantages'].shape == (10,)
        print("✓ PPOBuffer works")
        
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False
    
    return True

def test_ppo_agent():
    """Test PPO agent"""
    print("\nTesting PPO agent...")
    
    try:
        from tinyrl.agents.ppo import PPOAgent
        
        config = {
            "state_dim": 4,
            "action_dim": 2,
            "continuous_actions": False,
            "hidden_dim": 64,
            "attention_type": "linear",  # Use linear for faster testing
            "use_attention": True,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 1,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "rollout_length": 32,  # Small buffer for testing
            "device": "cpu"
        }
        
        agent = PPOAgent(config)
        
        # Test action selection
        state = np.random.randn(4)
        action = agent.select_action(state, training=True)
        assert isinstance(action, np.ndarray), f"Expected numpy array, got {type(action)}"
        assert action.shape == (), f"Expected scalar action, got shape {action.shape}"
        print("✓ PPO action selection works")
        
        # Test training step
        agent.step(1.0, False)
        print("✓ PPO step works")
        
        # Fill buffer and test update
        for i in range(35):  # Fill buffer
            state = np.random.randn(4)
            action = agent.select_action(state, training=True)
            agent.step(np.random.randn(), i == 34)
        
        if agent.buffer.is_full():
            update_info = agent.update()
            assert isinstance(update_info, dict), "Update should return dict"
            print("✓ PPO update works")
        
    except Exception as e:
        print(f"✗ PPO agent test failed: {e}")
        return False
    
    return True

def test_logger():
    """Test logger"""
    print("\nTesting logger...")
    
    try:
        from tinyrl.utils.logger import Logger
        
        logger = Logger("./test_logs")
        
        # Test logging
        metrics = {"reward": 10.5, "loss": 0.1}
        logger.log(metrics, step=1)
        
        # Test stats
        logger.log({"reward": 15.0}, step=2)
        stats = logger.get_stats("reward")
        assert "reward_mean" in stats
        print("✓ Logger works")
        
        logger.close()
        
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("TinyRL Framework Test Suite")
    print("=" * 50)
    
    tests = [
        test_attention_models,
        test_networks,
        test_replay_buffer,
        test_ppo_agent,
        test_logger,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 