#!/usr/bin/env python3
"""
MCP Agent Sandbox Demo Runner Script

This script demonstrates the core capabilities of the MCP Agent Sandbox framework:
1. Single LLM in one sandbox - Basic code execution  
2. One LLM controlling two sandboxes - Multi-environment coordination
3. Two LLMs in one sandbox - Collaborative problem solving

Usage:
    python run_demo.py
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Main demo function"""
    print("🌟 Welcome to MCP Agent Sandbox Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("✓ Multi-model LLM support (GPT, Claude, Gemini, Mistral, Llama)")
    print("✓ Isolated sandbox execution environments")
    print("✓ Single LLM workflows")
    print("✓ Multi-sandbox coordination")
    print("✓ Multi-agent collaboration")
    print("✓ Model Context Protocol (MCP) creation")
    print()
    
    try:
        # Import and run demos
        from demos.demo_runner import MCPAgentDemoRunner
        
        runner = MCPAgentDemoRunner()
        results = await runner.run_all_demos()
        
        # Print summary
        runner.print_summary(results)
        
        print("\n🎉 Demo completed successfully!")
        print("The MCP Agent Sandbox framework is ready for use.")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return None

if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main())
    
    # Exit with appropriate code
    if results:
        print("\n✅ All demos completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed. Check error messages above.")
        sys.exit(1) 