"""
MCP Agent Sandbox - Multi-Model LLM Agent Framework with Sandbox Execution
========================================================================

A lightweight framework for building intelligent agents that can:
- Execute code in isolated sandbox environments
- Support multiple LLM models (GPT, Claude, Gemini, Mistral, Llama)
- Coordinate between different agents and sandboxes
- Create and manage Model Context Protocols (MCPs)

Key Components:
- Multi-Model LLM Interface: Support for 5+ major LLM providers
- Sandbox Manager: Isolated code execution with virtual environments
- Manager Agent: Central coordinator for task orchestration
- Web Agent: External information retrieval capabilities
- MCP Tools: Dynamic tool creation and validation

Demo Scenarios:
1. Single LLM in one sandbox - Basic code execution
2. One LLM controlling two sandboxes - Multi-environment coordination  
3. Two LLMs in one sandbox - Collaborative problem solving

Usage:
    from mcp_agent_sandbox import create_demo_runner
    
    # Run all demos
    runner = create_demo_runner()
    results = await runner.run_all_demos()
"""

__version__ = "0.1.0"
__author__ = "MCP Agent Sandbox Team"

# Graceful imports with fallbacks
try:
    from models.llm_interface import (
        MultiModelManager,
        ModelConfig,
        ModelFactory,
        create_model_manager_from_env
    )
    _models_available = True
except ImportError as e:
    print(f"Warning: Models module not available: {e}")
    _models_available = False

try:
    from agents.sandbox import (
        SandboxManager,
        SandboxConfig,
        Sandbox,
        SandboxResult
    )
    _sandbox_available = True
except ImportError as e:
    print(f"Warning: Sandbox module not available: {e}")
    _sandbox_available = False

try:
    from agents.manager_agent import ManagerAgent
    from agents.web_agent import WebAgent
    from agents.mcp_tools import MCPBrainstorming, ScriptGeneratingTool, CodeRunningTool
    _agents_available = True
except ImportError as e:
    print(f"Warning: Agents module not available: {e}")
    _agents_available = False

try:
    from demos.demo_runner import MCPAgentDemoRunner
    _demos_available = True
except ImportError as e:
    print(f"Warning: Demo module not available: {e}")
    _demos_available = False

# Export main components
__all__ = [
    # Core version info
    "__version__",
    "__author__",
    
    # Models
    "MultiModelManager",
    "ModelConfig", 
    "ModelFactory",
    "create_model_manager_from_env",
    
    # Sandbox
    "SandboxManager",
    "SandboxConfig",
    "Sandbox",
    "SandboxResult",
    
    # Agents
    "ManagerAgent",
    "WebAgent",
    "MCPBrainstorming",
    "ScriptGeneratingTool", 
    "CodeRunningTool",
    
    # Demos
    "MCPAgentDemoRunner",
    
    # Convenience functions
    "create_demo_runner",
    "create_mcp_agent",
    "run_mcp_demos",
    "check_dependencies"
]

def create_demo_runner():
    """Create a demo runner for showcasing MCP Agent Sandbox capabilities"""
    if not _demos_available:
        raise ImportError("Demo runner not available. Please install required dependencies.")
    
    return MCPAgentDemoRunner()

async def create_mcp_agent(model_manager=None):
    """Create a fully configured MCP agent system"""
    try:
        if not _models_available or not _agents_available:
            raise ImportError("Agent components not available")
        
        if model_manager is None:
            model_manager = create_model_manager_from_env()
        
        # Import here to avoid circular imports
        from agents.manager_agent import create_manager_agent
        return await create_manager_agent(model_manager)
    except Exception as e:
        print(f"Warning: Could not create full MCP agent: {e}")
        print("Some dependencies may be missing. Please check requirements.txt")
        return None

async def run_mcp_demos():
    """Run all MCP Agent Sandbox demos"""
    try:
        runner = create_demo_runner()
        return await runner.run_all_demos()
    except Exception as e:
        print(f"Could not run demos: {e}")
        return []

def check_dependencies():
    """Check which dependencies are available"""
    status = {
        "models": _models_available,
        "sandbox": _sandbox_available, 
        "agents": _agents_available,
        "demos": _demos_available
    }
    
    print("MCP Agent Sandbox Dependency Status:")
    for component, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component.capitalize()}")
    
    if all(status.values()):
        print("\n🎉 All components available! Ready to run.")
    else:
        print("\n⚠️  Some components missing. Install with: pip install -r requirements.txt")
    
    return status

# Auto-check dependencies on import
if __name__ != "__main__":
    # Only show this during interactive use, not during testing
    import os
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        missing_count = sum(1 for x in [_models_available, _sandbox_available, _agents_available, _demos_available] if not x)
        if missing_count > 0:
            print(f"MCP Agent Sandbox: {missing_count} component(s) not available. Run check_dependencies() for details.") 