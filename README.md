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

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+
- Virtual environment manager (conda recommended, or venv)
- Git

### 🔧 Environment Setup & Installation

Choose one of the following methods:

#### Method 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/NoakLiu/TinyRL.git
cd TinyRL

# Create conda environment with Python 3.9+
conda create -n mcp_sandbox python=3.10 -y

# Activate conda environment
conda activate mcp_sandbox

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sys; print(f'Python: {sys.version}')"
```

### 🔑 Configure API Keys
Create a `.env` file in the project root and add your API keys:
```env
# Configure at least one LLM API key
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
MISTRAL_API_KEY=your-mistral-key-here

# Optional: Local Llama models
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Sandbox settings
MAX_CONCURRENT_SANDBOXES=5
SANDBOX_TIMEOUT=60
SANDBOX_MAX_MEMORY_MB=1024
```

## 🆓 Using Free Open Source Models

**Good news!** You can use completely free open source models without any API keys:

### Option 1: Ollama (Recommended for beginners)

#### Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull free models
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull codellama:7b
```

#### Update your .env file
```env
# No API keys needed!
OLLAMA_BASE_URL=http://localhost:11434
```

#### Test with free models
```python
# Test Qwen 2.5 (great for coding)
python -c "
import asyncio
from models.llm_interface import MultiModelManager

async def test():
    manager = MultiModelManager()
    response = await manager.generate(
        'Write a Python function to calculate fibonacci numbers',
        model_name='qwen2.5-coder'
    )
    print(response)

asyncio.run(test())
"
```

### Option 2: Hugging Face Models (Direct GPU/CPU)

#### Install additional dependencies
```bash
# Install PyTorch and Transformers
pip install torch transformers accelerate bitsandbytes
```

#### Configure for your hardware
```env
# Hardware configuration
HF_DEVICE=auto  # auto, cpu, cuda
HF_LOAD_IN_8BIT=true  # Save memory on GPU
```

#### Available free models
```python
# Test GPT-2 (small, fast, completely free)
response = await manager.generate(
    "Explain Python functions",
    model_name="gpt2"
)

# Test Qwen 2.5 7B (better quality, needs more memory)
response = await manager.generate(
    "Write a sorting algorithm",
    model_name="qwen-7b"
)

# Test Code Llama (specialized for coding)
response = await manager.generate(
    "Create a REST API in Python",
    model_name="codellama-7b"
)
```

### Option 3: Local API Servers

#### Using vLLM (for high performance)
```bash
# Install vLLM
pip install vllm

# Start server with Qwen model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# Configure in .env
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 🎯 Recommended Free Models

| Model | Best For | Memory | Speed |
|-------|----------|---------|-------|
| `gpt2` | Learning, testing | 500MB | ⚡⚡⚡ |
| `qwen2.5-coder` | Code generation | 4GB | ⚡⚡ |
| `qwen-7b` | General tasks | 7GB | ⚡ |
| `codellama-7b` | Code understanding | 7GB | ⚡ |

### 🔧 Memory Optimization Tips

```python
# For limited GPU memory
ModelConfig(
    model_name="qwen-7b",
    device="auto",
    load_in_8bit=True,  # Reduces memory by ~50%
    max_tokens=1000     # Limit output length
)

# For CPU-only systems
ModelConfig(
    model_name="gpt2",
    device="cpu",
    max_tokens=512
)
```

### ⚡ Quick Start with Free Models

```bash
# 1. Install Ollama models (easiest)
ollama pull qwen2.5-coder:7b

# 2. Test immediately
python -c "
import asyncio
from demos.demo_runner import MCPAgentDemoRunner

async def test_free():
    runner = MCPAgentDemoRunner()
    # This will now use free Qwen models!
    await runner.run_demo_1()

asyncio.run(test_free())
"
```

## 🚀 Running the Demo

```bash
# Run all three demo scenarios
python run_demo.py
```

## 🎯 After Running the Demo Successfully

Congratulations! If you see all demos completed successfully, you're ready for the next steps.

### 🔑 Get Real LLM Responses (Instead of Mock)

The demo uses mock responses by default. To get real AI responses, configure your API keys:

#### Step 1: Get API Keys
- **OpenAI**: Visit [OpenAI API](https://platform.openai.com/api-keys)
- **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com/)
- **Google**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Mistral**: Visit [Mistral AI Console](https://console.mistral.ai/)

#### Step 2: Update Your .env File
```env
# Replace with your actual API keys
OPENAI_API_KEY=sk-your-real-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-real-anthropic-key-here
GOOGLE_AI_STUDIO_API_KEY=your-real-google-key-here
MISTRAL_API_KEY=your-real-mistral-key-here
```

#### Step 3: Run Demo with Real APIs
```bash
# Activate your environment
conda activate mcp_sandbox

# Run demo with real API responses
python run_demo.py
```

### 🛠️ Start Building Your Own Applications

Now that the framework is working, here are some practical examples:

#### Example 1: Data Analysis Assistant
```python
import asyncio
from models.llm_interface import MultiModelManager
from agents.sandbox import SandboxManager

async def analyze_data():
    # Create managers
    model_manager = MultiModelManager()
    sandbox_manager = SandboxManager()
    
    # Create sandbox
    sandbox = await sandbox_manager.create_sandbox()
    
    # Generate analysis code
    prompt = """
    Create a Python script that:
    1. Generates sample sales data
    2. Performs statistical analysis
    3. Creates visualizations
    4. Saves results to files
    """
    
    response = await model_manager.generate(prompt, model_name="gpt-4o")
    
    # Execute in sandbox
    result = await sandbox.execute_code(response)
    print("Analysis completed:", result.success)
    
    # Cleanup
    await sandbox_manager.cleanup_sandbox(sandbox.id)

# Run the analysis
asyncio.run(analyze_data())
```

#### Example 2: Web Development Assistant
```python
async def create_web_app():
    model_manager = MultiModelManager()
    sandbox_manager = SandboxManager()
    
    sandbox = await sandbox_manager.create_sandbox()
    
    prompt = """
    Create a simple Flask web application that:
    1. Has a homepage with a form
    2. Processes form data
    3. Returns results as JSON
    4. Includes basic error handling
    """
    
    code = await model_manager.generate(prompt, model_name="claude-3.5-sonnet")
    result = await sandbox.execute_code(code)
    
    print("Web app created:", result.success)
    await sandbox_manager.cleanup_sandbox(sandbox.id)

asyncio.run(create_web_app())
```

### 📊 Monitor Your Usage

Track your API usage and performance:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor execution time
import time
start = time.time()

# Your code here
result = await model_manager.generate("Your prompt", model_name="gpt-4o")

execution_time = time.time() - start
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Token usage: {result.usage if hasattr(result, 'usage') else 'N/A'}")
```

### 🎛️ Custom Configuration

Create your own configuration for specific use cases:

```python
from models.llm_interface import ModelConfig

# Custom model configuration
custom_config = ModelConfig(
    model_name="gpt-4o",
    temperature=0.3,  # More deterministic
    max_tokens=2000,  # Longer responses
    timeout=30        # 30 second timeout
)

# Add to manager
model_manager.add_model("my-custom-gpt", custom_config)

# Use custom model
response = await model_manager.generate(
    "Complex analysis task", 
    model_name="my-custom-gpt"
)
```

### 📋 Quick Reference Commands

Here are the most common commands you'll use:

#### Basic Operations
```bash
# Activate environment
conda activate mcp_sandbox

# Run full demo
python run_demo.py

# Check system status
python -c "from __init__ import check_dependencies; check_dependencies()"
```

#### Interactive Python Session
```python
# Quick start template
import asyncio
from models.llm_interface import MultiModelManager
from agents.sandbox import SandboxManager

async def main():
    # Setup
    model_manager = MultiModelManager()
    sandbox_manager = SandboxManager()
    sandbox = await sandbox_manager.create_sandbox()
    
    # Your code here
    response = await model_manager.generate("Your prompt", model_name="gpt-4o")
    result = await sandbox.execute_code(response)
    
    print("Success:", result.success)
    print("Output:", result.output)
    
    # Cleanup
    await sandbox_manager.cleanup_sandbox(sandbox.id)

# Run it
asyncio.run(main())
```

#### Common Use Cases
```python
# 1. Code Generation & Execution
prompt = "Create a Python function to calculate fibonacci numbers"
code = await model_manager.generate(prompt, model_name="gpt-4o")
result = await sandbox.execute_code(code)

# 2. Data Analysis
prompt = "Analyze this CSV data and create visualizations"
analysis_code = await model_manager.generate(prompt, model_name="claude-3.5-sonnet")
analysis_result = await sandbox.execute_code(analysis_code)

# 3. Multi-step Tasks
step1 = await model_manager.generate("Generate sample data", model_name="gpt-4o")
step2 = await model_manager.generate("Process the data from step 1", model_name="claude-3.5-sonnet")
```

### 🔧 Advanced Configuration

#### Environment Variables
```bash
# Performance tuning
export MAX_CONCURRENT_SANDBOXES=3
export SANDBOX_TIMEOUT=60
export ENABLE_RESPONSE_CACHE=true

# Logging
export LOG_LEVEL=INFO
export ENABLE_DEBUG_MODE=false
```

#### Custom Sandbox Configuration
```python
from agents.sandbox import SandboxConfig

custom_sandbox = SandboxConfig(
    timeout=45,           # 45 second timeout
    max_memory_mb=1024,   # 1GB memory limit
    enable_network=True,  # Allow internet access
    python_packages=["numpy", "pandas", "matplotlib"]  # Pre-install packages
)

sandbox = await sandbox_manager.create_sandbox(config=custom_sandbox)
```

## 📋 Complete Running Guide

### 🎮 Interactive Usage Examples

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
# Create conda activation script
cat > quick_run.sh << 'EOF'
#!/bin/bash
conda activate mcp_sandbox
python run_demo.py
EOF

chmod +x quick_run.sh
./quick_run.sh
```

### 📚 Learning Path Recommendations

1. **Beginners**: Start with `python run_demo.py` to understand basic functionality
2. **Intermediate**: Read `demos/demo_runner.py` to understand implementation details
3. **Developers**: Explore `agents/` and `models/` directories to learn architecture
4. **Customization**: Modify configurations and create custom agents based on your needs

### 🚀 Quick Test Commands

```bash
# Activate environment first
conda activate mcp_sandbox  # or: source mcp_env/bin/activate

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
