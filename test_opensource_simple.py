#!/usr/bin/env python3
"""
Simple Open Source Models Test

Tests only open source models without commercial API dependencies.
"""

import asyncio
import sys
import os
import requests
import json

async def test_gpt2_direct():
    """Test GPT-2 directly with transformers"""
    print("🔧 Testing GPT-2 (free, small, fast)...")
    try:
        from transformers import pipeline
        
        # Create a simple text generation pipeline
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device="cpu",  # Use CPU for compatibility
            max_length=150
        )
        
        prompt = "def fibonacci(n):"
        outputs = generator(
            prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        response = outputs[0]["generated_text"]
        print(f"✅ GPT-2 Response:\n{response}")
        return True
        
    except Exception as e:
        print(f"❌ GPT-2 Error: {e}")
        print("💡 Try: pip install torch transformers")
        return False

async def test_qwen_via_ollama():
    """Test Qwen via Ollama API directly"""
    print("\n🔧 Testing Qwen 2.5 Coder via Ollama...")
    try:
        base_url = "http://localhost:11434"
        
        # Format for Qwen models
        prompt = "Write a Python function to calculate factorial:"
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": "qwen2.5-coder",
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200,
                "stop": ["<|im_end|>"]
            }
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()["response"].strip()
            print(f"✅ Qwen Response:\n{result}")
            return True
        else:
            print(f"❌ Ollama API Error: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Qwen/Ollama Error: {e}")
        print("💡 Install Ollama and run: ollama pull qwen2.5-coder")
        return False

def check_simple_dependencies():
    """Check basic dependencies"""
    print("🔍 Checking dependencies...")
    
    deps = {
        "transformers": False,
        "torch": False,
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
    
    return deps

async def run_simple_tests():
    """Run simple tests"""
    print("🌟 Simple Open Source Model Testing")
    print("=" * 50)
    
    deps = check_simple_dependencies()
    
    print("\n📋 Running Available Tests...\n")
    
    results = {}
    
    # Test GPT-2 if transformers is available
    if deps["transformers"] and deps["torch"]:
        results["gpt2"] = await test_gpt2_direct()
    else:
        print("⏭️ Skipping GPT-2 (transformers/torch not available)")
    
    # Test Ollama if available
    if deps["ollama"]:
        results["ollama"] = await test_qwen_via_ollama()
    else:
        print("⏭️ Skipping Ollama (server not running)")
    
    return results

def print_simple_guide():
    """Print simple installation guide"""
    print("\n" + "=" * 60)
    print("📚 QUICK SETUP GUIDE")
    print("=" * 60)
    
    print("\n🚀 Option 1: Install PyTorch + Transformers")
    print("pip install torch transformers")
    print("# Then run this script to test GPT-2")
    
    print("\n🚀 Option 2: Install Ollama (Recommended)")
    print("# On Linux/Mac:")
    print("curl -fsSL https://ollama.ai/install.sh | sh")
    print("ollama pull qwen2.5-coder")
    print("# Then run this script to test Qwen")
    
    print("\n🚀 Option 3: Install both for maximum compatibility")
    print("pip install torch transformers")
    print("curl -fsSL https://ollama.ai/install.sh | sh")
    print("ollama pull qwen2.5-coder")

async def main():
    """Main test function"""
    print("🤖 Simple Open Source Model Test")
    print("Testing free AI models without commercial API dependencies!")
    print("-" * 50)
    
    try:
        results = await run_simple_tests()
        
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
            print("   You can now use free AI models!")
            
            if "gpt2" in results and results["gpt2"]:
                print("\n💡 GPT-2 Usage Example:")
                print("from transformers import pipeline")
                print("generator = pipeline('text-generation', model='gpt2')")
                print("output = generator('Your prompt here')")
            
            if "ollama" in results and results["ollama"]:
                print("\n💡 Ollama Usage Example:")
                print("curl -X POST http://localhost:11434/api/generate \\")
                print("  -H 'Content-Type: application/json' \\")
                print("  -d '{\"model\": \"qwen2.5-coder\", \"prompt\": \"Your prompt\"}'")
        
        print_simple_guide()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        print_simple_guide()

if __name__ == "__main__":
    # Run the simple tests
    asyncio.run(main()) 