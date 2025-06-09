#!/usr/bin/env python3
"""
Test Open Source Models Script

This script helps you test various free open source models
without needing any API keys.
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_gpt2():
    """Test GPT-2 model (smallest, fastest)"""
    print("🔧 Testing GPT-2 (free, small, fast)...")
    try:
        from models.llm_interface import MultiModelManager, ModelConfig
        
        manager = MultiModelManager()
        manager.add_model("gpt2", ModelConfig(
            model_name="gpt2",
            api_key="",
            device="cpu",  # Use CPU for compatibility
            max_tokens=100
        ))
        
        response = await manager.generate(
            "Write a simple Python function:",
            model_name="gpt2"
        )
        
        print(f"✅ GPT-2 Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ GPT-2 Error: {e}")
        print("💡 Try: pip install torch transformers")
        return False

async def test_ollama_qwen():
    """Test Qwen via Ollama"""
    print("\n🔧 Testing Qwen 2.5 Coder via Ollama...")
    try:
        from models.llm_interface import MultiModelManager, ModelConfig
        
        manager = MultiModelManager()
        manager.add_model("qwen-coder", ModelConfig(
            model_name="qwen2.5-coder",
            api_key="",
            base_url="http://localhost:11434"
        ))
        
        response = await manager.generate(
            "Write a Python function to calculate factorial:",
            model_name="qwen-coder"
        )
        
        print(f"✅ Qwen Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen/Ollama Error: {e}")
        print("💡 Install Ollama and run: ollama pull qwen2.5-coder")
        return False

async def test_huggingface_qwen():
    """Test Qwen via Hugging Face"""
    print("\n🔧 Testing Qwen 2.5 via Hugging Face...")
    try:
        from models.llm_interface import MultiModelManager, ModelConfig
        
        manager = MultiModelManager()
        manager.add_model("qwen-hf", ModelConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            api_key="",
            device="auto",
            load_in_8bit=True,
            max_tokens=200
        ))
        
        response = await manager.generate(
            "Explain what Python is in one sentence:",
            model_name="qwen-hf"
        )
        
        print(f"✅ Qwen HF Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen HF Error: {e}")
        print("💡 This requires significant memory (4GB+)")
        return False

def check_dependencies():
    """Check what dependencies are available"""
    print("🔍 Checking dependencies...")
    
    deps = {
        "transformers": False,
        "torch": False,
        "requests": False,
        "ollama": False
    }
    
    try:
        import transformers
        deps["transformers"] = True
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers: Not installed")
    
    try:
        import torch
        deps["torch"] = True
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ torch: Not installed")
    
    try:
        import requests
        # Test Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            deps["ollama"] = True
            models = response.json().get("models", [])
            print(f"✅ ollama: Running with {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']}")
        else:
            print("❌ ollama: Server not responding")
    except Exception:
        print("❌ ollama: Not running or not installed")
    
    deps["requests"] = True
    return deps

async def run_available_tests():
    """Run tests for available models"""
    print("🌟 Open Source Model Testing")
    print("=" * 50)
    
    deps = check_dependencies()
    
    print("\n📋 Running Available Tests...\n")
    
    results = {}
    
    # Test GPT-2 if transformers is available
    if deps["transformers"] and deps["torch"]:
        results["gpt2"] = await test_gpt2()
    else:
        print("⏭️ Skipping GPT-2 (transformers/torch not available)")
    
    # Test Ollama if available
    if deps["ollama"]:
        results["ollama"] = await test_ollama_qwen()
    else:
        print("⏭️ Skipping Ollama (server not running)")
    
    # Test Hugging Face Qwen if dependencies available
    if deps["transformers"] and deps["torch"]:
        print("\n⚠️ Hugging Face model test requires significant memory...")
        user_input = input("Do you want to test Qwen 7B? (y/N): ")
        if user_input.lower() in ['y', 'yes']:
            results["huggingface"] = await test_huggingface_qwen()
    
    return results

def print_installation_guide():
    """Print installation guide for different options"""
    print("\n" + "=" * 60)
    print("📚 INSTALLATION GUIDE FOR OPEN SOURCE MODELS")
    print("=" * 60)
    
    print("\n🎯 Option 1: Ollama (Easiest)")
    print("curl -fsSL https://ollama.ai/install.sh | sh")
    print("ollama pull qwen2.5-coder:7b")
    print("ollama pull llama3.1:8b")
    
    print("\n🎯 Option 2: Hugging Face (More control)")
    print("pip install torch transformers accelerate bitsandbytes")
    
    print("\n🎯 Option 3: vLLM (High performance)")
    print("pip install vllm")
    print("python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct")
    
    print("\n💡 Memory Requirements:")
    print("- GPT-2: 500MB RAM")
    print("- Qwen 2.5 7B: 4GB+ RAM/VRAM")
    print("- CodeLlama 7B: 4GB+ RAM/VRAM")

async def main():
    """Main test function"""
    print("🤖 Open Source Model Test Suite")
    print("Testing free AI models without API keys!")
    print("-" * 50)
    
    try:
        results = await run_available_tests()
        
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        success_count = sum(1 for success in results.values() if success)
        total_tests = len(results)
        
        if total_tests > 0:
            print(f"✅ Successful: {success_count}/{total_tests}")
            for model, success in results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   {model}: {status}")
        else:
            print("❌ No tests could be run.")
            print("   Please install dependencies first.")
        
        if success_count > 0:
            print(f"\n🎉 Great! You have {success_count} working model(s)!")
            print("   You can now use the MCP Agent Sandbox with free models!")
        
        print_installation_guide()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        print_installation_guide()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main()) 