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
