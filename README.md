# 🤖 Tiny RL MCP Agent Sandbox

**Multi-Model LLM Agent Framework with Sandbox Execution**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

A lightweight, powerful framework for building intelligent agents that execute code in isolated sandbox environments while supporting multiple LLM models and enabling collaborative multi-agent workflows.

## ✨ Key Features

- 🎯 **Multi-Model LLM Support**: Seamlessly integrate with GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Mistral-Large, and Llama-3.1
- 🔒 **Isolated Sandbox Execution**: Safe code execution in virtual environments with automatic cleanup
- 🤝 **Multi-Agent Collaboration**: Support for complex agent interactions and task coordination
- 🧠 **Dynamic Tool Creation**: Automatic Model Context Protocol (MCP) generation and management
- 🌐 **Web Integration**: Built-in web search and information retrieval capabilities
- ⚡ **Async/Parallel Processing**: High-performance concurrent operations
- 🛡️ **Error Recovery**: Robust error handling and automatic retry mechanisms

## 🎭 Demo Scenarios

### 1. 🚀 Single LLM + Single Sandbox
**Basic code execution with one model in an isolated environment**
- Perfect for simple computational tasks
- Demonstrates basic LLM-to-code translation
- Shows sandbox isolation and safety features

### 2. 🔄 Single LLM + Dual Sandbox  
**Coordinated multi-environment execution**
- Data retrieval in one sandbox, processing in another
- Simulates real-world distributed computing patterns
- Demonstrates inter-sandbox communication

### 3. 🤝 Dual LLM + Single Sandbox
**Multi-agent collaboration in shared environment**
- Planning agent + Implementation agent cooperation
- Complex problem decomposition and solving
- Shows emergent collaborative behaviors

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/NoakLiu/TinyRL.git
cd TinyRL

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Run all three demo scenarios
python run_demo.py
```

## 📋 Complete Running Guide

### 1. 🔧 Environment Setup

#### Step 1: Check Python Version
```bash
python --version  # Requires Python 3.9 or higher
```

#### Step 2: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv mcp_env

# Activate virtual environment (Linux/Mac)
source mcp_env/bin/activate

# Activate virtual environment (Windows)
mcp_env\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
# Install all required dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sys; print(f'Python: {sys.version}')"
```

#### Step 4: Configure API Keys
Create a `.env` file and add your API keys:
```env
# Configure at least one LLM API key
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
MISTRAL_API_KEY=your-mistral-key-here

# Optional: Local Llama models
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. 🚀 Basic Running Methods

#### Method 1: Run Complete Demo
```bash
# Run all three demo scenarios
python run_demo.py
```

#### Method 2: Interactive Python Usage
```python
# Start Python interpreter
python

# Run in Python
import asyncio
from demos.demo_runner import MCPAgentDemoRunner

# Create demo runner
runner = MCPAgentDemoRunner()

# Run all demos
results = asyncio.run(runner.run_all_demos())

# View results
runner.print_summary(results)
```

#### Method 3: Check System Status
```python
# Check dependency status
python -c "
import sys
sys.path.insert(0, '.')
from __init__ import check_dependencies
check_dependencies()
"
```

### 3. 🎯 Advanced Usage Methods

#### Run Specific Demos Individually
```python
import asyncio
from demos.demo_runner import MCPAgentDemoRunner

async def run_specific_demo():
    runner = MCPAgentDemoRunner()
    
    # Run Demo 1: Single LLM + Single Sandbox
    result1 = await runner.run_demo_1()
    
    # Run Demo 2: Single LLM + Dual Sandbox
    result2 = await runner.run_demo_2()
    
    # Run Demo 3: Dual LLM + Single Sandbox
    result3 = await runner.run_demo_3()
    
    return [result1, result2, result3]

# Execute specific demo
asyncio.run(run_specific_demo())
```

#### Custom Model Configuration
```python
from models.llm_interface import MultiModelManager, ModelConfig

# Create custom model manager
manager = MultiModelManager()

# Add specific model configuration
manager.add_model("my-gpt", ModelConfig(
    model_name="gpt-4o",
    api_key="your-key-here",
    temperature=0.7,
    max_tokens=4096
))

# Use custom model
response = await manager.generate(
    prompt="Explain the fundamentals of quantum computing",
    model_name="my-gpt"
)
```

#### Create Custom Agents
```python
from agents.manager_agent import ManagerAgent
from agents.sandbox import SandboxManager

# Create sandbox manager
sandbox_manager = SandboxManager()

# Create manager agent
agent = ManagerAgent(model_manager, sandbox_manager)

# Execute specific task
result = await agent.process_task(
    "Analyze the given dataset and generate visualization charts"
)
```

### 4. 🐛 Troubleshooting Guide

#### Common Issues and Solutions

**Issue 1: Import Errors**
```bash
# Symptom: ImportError or ModuleNotFoundError
# Solution:
python -c "import sys; print(sys.path)"
pip install -r requirements.txt --force-reinstall
```

**Issue 2: API Key Errors**
```bash
# Symptom: Authentication failed
# Solutions:
# 1. Check if .env file exists
ls -la .env

# 2. Verify environment variables
python -c "import os; print('OPENAI_API_KEY' in os.environ)"

# 3. Reset environment variables
export OPENAI_API_KEY="your-actual-key-here"
```

**Issue 3: Sandbox Creation Failed**
```bash
# Symptom: Sandbox creation failed
# Solutions:
# 1. Check disk space
df -h

# 2. Verify Python version
python --version

# 3. Test virtual environment support
python -m venv test_env && rm -rf test_env
```

**Issue 4: Slow Demo Execution**
```python
# Symptom: Demo takes too long to run
# Solution: Adjust configuration parameters
from agents.sandbox import SandboxConfig

config = SandboxConfig(
    timeout=30,  # Reduce timeout
    max_memory_mb=512,  # Limit memory usage
    enable_network=False  # Disable network access
)
```

### 5. 📊 Performance Optimization

#### Basic Optimization
```python
# 1. Limit concurrent sandboxes
import os
os.environ["MAX_CONCURRENT_SANDBOXES"] = "3"

# 2. Optimize model parameters
model_config = ModelConfig(
    model_name="gpt-4o-mini",  # Use faster model
    max_tokens=1000,  # Limit output length
    temperature=0.1   # Reduce randomness
)

# 3. Enable caching
os.environ["ENABLE_RESPONSE_CACHE"] = "true"
```

#### Monitoring and Debugging
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Performance monitoring
import time
start_time = time.time()
# ... run your code ...
print(f"Execution time: {time.time() - start_time:.2f} seconds")
```

### 6. 🎮 Interactive Usage Examples

#### Jupyter Notebook Usage
```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

import asyncio
from demos.demo_runner import MCPAgentDemoRunner

# Create async loop
runner = MCPAgentDemoRunner()
results = await runner.run_all_demos()

# Display results
for i, result in enumerate(results, 1):
    print(f"Demo {i} Result:")
    print(result)
    print("-" * 50)
```

#### Command Line Shortcuts
```bash
# Create run script
cat > quick_run.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python run_demo.py
EOF

chmod +x quick_run.sh
./quick_run.sh
```

### 7. 📚 Learning Path Recommendations

1. **Beginners**: Start with `python run_demo.py` to understand basic functionality
2. **Intermediate**: Read `demos/demo_runner.py` to understand implementation details
3. **Developers**: Explore `agents/` and `models/` directories to learn architecture
4. **Customization**: Modify configurations and create custom agents based on your needs

### 8. 🔍 Verify Installation Success

Run the following commands to verify everything is working:
```bash
# Complete verification script
python -c "
import sys
print('Python Version:', sys.version)

# Check main modules
try:
    from demos.demo_runner import MCPAgentDemoRunner
    print('✅ Demo Module: OK')
except ImportError as e:
    print('❌ Demo Module:', e)

try:
    from models.llm_interface import MultiModelManager
    print('✅ Model Interface: OK')
except ImportError as e:
    print('❌ Model Interface:', e)

try:
    from agents.sandbox import SandboxManager
    print('✅ Sandbox Manager: OK')
except ImportError as e:
    print('❌ Sandbox Manager:', e)

print('🎉 Verification Complete!')
"
```

If all checks pass, you're ready to start using MCP Agent Sandbox!

### 9. 🚀 Quick Test Commands

```bash
# Test 1: Basic functionality
python -c "from __init__ import check_dependencies; check_dependencies()"

# Test 2: Run single demo
python -c "
import asyncio
from demos.demo_runner import MCPAgentDemoRunner
runner = MCPAgentDemoRunner()
result = asyncio.run(runner.run_demo_1())
print('Demo 1 completed:', 'Success' if result else 'Failed')
"

# Test 3: Model interface
python -c "
from models.llm_interface import MultiModelManager
manager = MultiModelManager()
print('Model manager created successfully')
"
```

### Basic Usage

```python
import asyncio
from mcp_agent_sandbox import create_demo_runner, create_mcp_agent

# Run demos
async def main():
    # Quick demo
    runner = create_demo_runner()
    results = await runner.run_all_demos()
    
    # Create your own agent
    agent = await create_mcp_agent()
    result = await agent.process_task("Analyze this dataset and create visualizations")
    print(result)

asyncio.run(main())
```

## 🏗️ Architecture

```
MCP Agent Sandbox
├── 🧠 Manager Agent      # Central coordinator
├── 🌐 Web Agent          # Information retrieval  
├── 🔧 MCP Tools          # Dynamic tool creation
├── 📦 Sandbox Manager    # Isolated execution
└── 🤖 Multi-Model LLMs   # AI reasoning engine
```

### Core Components

- **Manager Agent**: Implements minimal predefinition and maximal self-evolution principles
- **Web Agent**: Handles external information retrieval and web browsing
- **Sandbox Manager**: Provides secure, isolated code execution environments
- **MCP Tools**: Creates, validates, and manages Model Context Protocols dynamically
- **Multi-Model Interface**: Unified API for different LLM providers

## 🎯 Use Cases

### 🔬 Research & Analysis
- Automated data analysis pipelines
- Literature review and synthesis
- Experimental design and execution

### 💻 Software Development
- Code generation and testing
- Multi-language project scaffolding
- Automated debugging and optimization

### 📊 Business Intelligence
- Report generation and insights
- Process automation
- Decision support systems

### 🎓 Education & Training
- Interactive coding tutorials
- Automated assignment grading
- Personalized learning paths

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM API Keys (add the ones you want to use)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
MISTRAL_API_KEY=your_mistral_key_here

# Optional: Custom endpoints
OLLAMA_BASE_URL=http://localhost:11434  # For local Llama models

# Sandbox Settings
MAX_CONCURRENT_SANDBOXES=5
SANDBOX_TIMEOUT=60
SANDBOX_MAX_MEMORY_MB=1024
```

### Model Configuration

```python
from mcp_agent_sandbox import MultiModelManager, ModelConfig

# Setup your preferred models
manager = MultiModelManager()

# Add models individually
manager.add_model("my-gpt", ModelConfig(
    model_name="gpt-4o",
    api_key="your-key",
    temperature=0.7,
    max_tokens=4096
))

# Or use environment-based setup
manager = create_model_manager_from_env()
```

## 📚 API Reference

### Core Classes

#### `MCPAgentDemoRunner`
```python
runner = MCPAgentDemoRunner()
results = await runner.run_all_demos()
```

#### `MultiModelManager`
```python
manager = MultiModelManager()
response = await manager.generate(
    prompt="Explain quantum computing",
    model_name="gpt-4o"
)
```

#### `SandboxManager`
```python
sandbox_manager = SandboxManager()
sandbox_id = await sandbox_manager.create_sandbox(config)
result = sandbox.execute_code("print('Hello, World!')")
```

### Utility Functions

```python
# Check system dependencies
from mcp_agent_sandbox import check_dependencies
status = check_dependencies()

# Quick demo runner
from mcp_agent_sandbox import run_mcp_demos
results = await run_mcp_demos()
```

## 🛡️ Security & Safety

- **Sandbox Isolation**: All code execution happens in isolated virtual environments
- **Resource Limits**: Configurable memory, CPU, and time constraints
- **Network Controls**: Optional network access restrictions
- **Automatic Cleanup**: Temporary files and environments are automatically removed
- **Error Boundaries**: Robust error handling prevents system-wide failures

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
flake8 .
```

## 📊 Performance

- **Startup Time**: < 2 seconds
- **Sandbox Creation**: ~100ms per sandbox
- **Code Execution**: Varies by complexity
- **Memory Usage**: ~50MB base + sandbox overhead
- **Concurrent Sandboxes**: Up to 10 recommended

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Check dependencies
python -c "from mcp_agent_sandbox import check_dependencies; check_dependencies()"

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

**API Key Issues**
```bash
# Verify environment variables
python -c "import os; print('OPENAI_API_KEY' in os.environ)"
```

**Sandbox Failures**
- Ensure Python 3.9+ is installed
- Check available disk space
- Verify virtual environment support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original Alita research paper on minimal predefinition and maximal self-evolution
- Built with modern Python async/await patterns
- Leverages the power of multiple LLM providers
- Special thanks to the open-source community

<!-- ## 🔗 Links

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](issues/)
- 💬 [Discussions](discussions/)
- 📦 [PyPI Package](https://pypi.org/project/mcp-agent-sandbox/) *(Coming Soon)* -->

---

**Made with ❤️ by the MCP Agent Sandbox Team**

*Empowering intelligent agents through minimal predefinition and maximal self-evolution*
