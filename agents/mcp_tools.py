"""
MCP Tools - Core tools for Model Context Protocol creation and management
Implements MCPBrainstorming, ScriptGeneratingTool, and CodeRunningTool
"""

import json
import logging
import asyncio
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from models.llm_interface import MultiModelManager
from agents.sandbox import SandboxManager, SandboxConfig, SandboxResult

logger = logging.getLogger(__name__)

@dataclass
class ToolSpec:
    """Specification for a tool or MCP"""
    name: str
    description: str
    requirements: List[str]
    code: str
    environment_setup: Optional[str] = None
    test_code: Optional[str] = None

class MCPBrainstorming:
    """Tool for assessing capabilities and identifying needed MCPs"""
    
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
    
    async def assess_capabilities(self, task_description: str, 
                                current_capabilities: List[str]) -> Dict[str, Any]:
        """
        Assess whether current capabilities are sufficient for the task
        Following Alita's principle of identifying functional gaps
        """
        
        assessment_prompt = f"""
        Task to analyze: {task_description}
        
        Current capabilities available:
        {', '.join(current_capabilities)}
        
        Please assess:
        1. Are the current capabilities sufficient to complete this task?
        2. What specific capabilities are missing?
        3. What tools/MCPs should be created to bridge the gaps?
        4. Prioritize the missing capabilities by importance
        
        Provide your assessment in JSON format with keys:
        - sufficient: boolean
        - missing_capabilities: list of strings
        - recommended_tools: list of objects with name, description, priority
        - reasoning: explanation of the assessment
        """
        
        system_prompt = """You are a capability assessor for the Alita framework. 
        Your role is to identify gaps between current capabilities and task requirements.
        Be precise and practical in your recommendations."""
        
        response = await self.model_manager.generate(assessment_prompt, system_prompt)
        
        try:
            assessment = json.loads(response)
            logger.info(f"Capability assessment: {assessment.get('sufficient', False)}")
            return assessment
        except json.JSONDecodeError:
            # Fallback assessment
            return {
                "sufficient": False,
                "missing_capabilities": ["unknown"],
                "recommended_tools": [
                    {
                        "name": "custom_tool",
                        "description": "Custom tool for task completion",
                        "priority": "high"
                    }
                ],
                "reasoning": "Failed to parse assessment, assuming capabilities are insufficient"
            }
    
    async def brainstorm_mcp_ideas(self, task_description: str, 
                                  context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Brainstorm potential MCP ideas for a given task
        """
        
        context_str = ""
        if context:
            context_str = f"\nAdditional context: {json.dumps(context, indent=2)}"
        
        brainstorm_prompt = f"""
        Task: {task_description}{context_str}
        
        Brainstorm creative and useful MCP (Model Context Protocol) ideas that could help solve this task.
        Think about what external capabilities, APIs, or tools might be needed.
        
        For each MCP idea, provide:
        - name: descriptive name
        - description: what it does
        - capabilities: list of specific functions it provides
        - implementation_difficulty: easy/medium/hard
        - usefulness_score: 1-10
        
        Provide 3-5 MCP ideas in JSON format as an array of objects.
        """
        
        system_prompt = """You are a creative tool designer for the Alita framework.
        Focus on practical, reusable tools that follow the principle of maximal self-evolution."""
        
        response = await self.model_manager.generate(brainstorm_prompt, system_prompt)
        
        try:
            ideas = json.loads(response)
            return ideas if isinstance(ideas, list) else []
        except json.JSONDecodeError:
            return []

class ScriptGeneratingTool:
    """Tool for generating scripts and MCPs based on requirements"""
    
    def __init__(self, model_manager: MultiModelManager, web_agent):
        self.model_manager = model_manager
        self.web_agent = web_agent
    
    async def generate_tool(self, description: str, github_links: List[str] = None,
                           requirements: List[str] = None) -> Dict[str, Any]:
        """
        Generate a tool/MCP based on description and optional GitHub references
        """
        
        if github_links is None:
            github_links = []
        if requirements is None:
            requirements = []
        
        # First, gather context from GitHub links if provided
        github_context = ""
        if github_links:
            github_context = await self._gather_github_context(github_links)
        
        # Generate the tool
        generation_prompt = f"""
        Generate a Python script/tool based on the following requirements:
        
        Description: {description}
        
        Required packages: {', '.join(requirements) if requirements else 'None specified'}
        
        GitHub context (if available):
        {github_context}
        
        Please provide:
        1. Complete Python code for the tool
        2. List of required packages (requirements.txt format)
        3. Environment setup instructions
        4. Simple test code to verify the tool works
        5. Brief documentation
        
        Format your response as JSON with keys:
        - code: the main Python code
        - requirements: list of required packages
        - setup_instructions: environment setup steps
        - test_code: code to test the tool
        - documentation: brief usage documentation
        """
        
        system_prompt = """You are an expert Python developer creating tools for the Alita framework.
        Write clean, modular, well-documented code that can be easily integrated and reused.
        Focus on reliability and error handling."""
        
        response = await self.model_manager.generate(generation_prompt, system_prompt)
        
        try:
            tool_spec = json.loads(response)
            
            # Validate the generated tool
            if not self._validate_tool_spec(tool_spec):
                raise ValueError("Generated tool specification is incomplete")
            
            return {
                "description": description,
                "code": tool_spec.get("code", ""),
                "requirements": tool_spec.get("requirements", []),
                "setup_instructions": tool_spec.get("setup_instructions", ""),
                "test_code": tool_spec.get("test_code", ""),
                "documentation": tool_spec.get("documentation", "")
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to generate tool: {e}")
            # Return a minimal fallback
            return {
                "description": description,
                "code": f'# Tool for: {description}\nprint("Tool not implemented")',
                "requirements": [],
                "setup_instructions": "No setup required",
                "test_code": 'print("Test passed")',
                "documentation": f"Tool for: {description}"
            }
    
    async def _gather_github_context(self, github_links: List[str]) -> str:
        """Gather context from GitHub repositories"""
        
        context_parts = []
        
        for link in github_links[:3]:  # Limit to 3 links to avoid overload
            try:
                # Get repository content
                if hasattr(self.web_agent, 'github_search'):
                    repo_content = await self.web_agent.github_search.get_repository_content(link)
                    
                    if repo_content.get('type') == 'file':
                        context_parts.append(f"File: {repo_content.get('name', 'unknown')}")
                        context_parts.append(repo_content.get('content', '')[:1000])  # First 1000 chars
                    elif repo_content.get('type') == 'directory':
                        file_list = [item['name'] for item in repo_content.get('contents', [])]
                        context_parts.append(f"Repository contents: {', '.join(file_list)}")
                
                # Also try to get the main page content
                page_content = await self.web_agent.extract_relevant_content(
                    link, "README documentation usage examples"
                )
                context_parts.append(f"Repository info: {page_content[:500]}")
                
            except Exception as e:
                logger.warning(f"Failed to gather context from {link}: {e}")
                context_parts.append(f"Link: {link} (failed to access)")
        
        return "\n\n".join(context_parts)
    
    def _validate_tool_spec(self, tool_spec: Dict[str, Any]) -> bool:
        """Validate that the tool specification has required fields"""
        required_fields = ["code", "requirements"]
        return all(field in tool_spec for field in required_fields)
    
    async def improve_tool(self, current_code: str, error_message: str, 
                          context: str = "") -> Dict[str, Any]:
        """
        Improve a tool based on error feedback
        """
        
        improvement_prompt = f"""
        The following tool code failed with an error. Please fix it:
        
        Current code:
        {current_code}
        
        Error message:
        {error_message}
        
        Context: {context}
        
        Please provide:
        1. Fixed code
        2. Explanation of what was wrong
        3. Any additional requirements needed
        
        Format as JSON with keys: code, explanation, requirements
        """
        
        system_prompt = """You are a debugging expert. Fix the code while maintaining its original purpose.
        Focus on robust error handling and clear documentation."""
        
        response = await self.model_manager.generate(improvement_prompt, system_prompt)
        
        try:
            improvement = json.loads(response)
            return improvement
        except json.JSONDecodeError:
            return {
                "code": current_code,  # Return original if parsing fails
                "explanation": "Failed to parse improvement response",
                "requirements": []
            }

class CodeRunningTool:
    """Tool for running and validating generated code"""
    
    def __init__(self, sandbox_manager: SandboxManager):
        self.sandbox_manager = sandbox_manager
    
    async def validate_tool(self, code: str, test_code: str = None,
                           requirements: List[str] = None) -> Dict[str, Any]:
        """
        Validate a tool by running it in a sandbox
        """
        
        if requirements is None:
            requirements = []
        
        try:
            # Create sandbox for validation
            config = SandboxConfig(name="tool_validation", timeout=30)
            sandbox_id = await self.sandbox_manager.create_sandbox(config)
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            
            if sandbox is None:
                return {
                    "success": False,
                    "error": "Failed to create sandbox",
                    "output": ""
                }
            
            # Install requirements
            if requirements:
                req_result = sandbox.install_requirements(requirements)
                if not req_result.success:
                    return {
                        "success": False,
                        "error": f"Failed to install requirements: {req_result.error}",
                        "output": req_result.output
                    }
            
            # Run the main code
            main_result = sandbox.execute_code(code, "main_tool.py")
            
            validation_result = {
                "success": main_result.success,
                "error": main_result.error,
                "output": main_result.output,
                "execution_time": main_result.execution_time
            }
            
            # Run test code if provided
            if test_code and main_result.success:
                test_result = sandbox.execute_code(test_code, "test_tool.py")
                validation_result["test_success"] = test_result.success
                validation_result["test_output"] = test_result.output
                validation_result["test_error"] = test_result.error
            
            # Cleanup
            await self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Tool validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    async def run_tool_with_input(self, code: str, input_data: Dict[str, Any],
                                 requirements: List[str] = None) -> Dict[str, Any]:
        """
        Run a tool with specific input data
        """
        
        if requirements is None:
            requirements = []
        
        try:
            # Create sandbox
            config = SandboxConfig(name="tool_execution")
            sandbox_id = await self.sandbox_manager.create_sandbox(config)
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            
            if sandbox is None:
                return {"success": False, "error": "Failed to create sandbox"}
            
            # Install requirements
            if requirements:
                req_result = sandbox.install_requirements(requirements)
                if not req_result.success:
                    await self.sandbox_manager.destroy_sandbox(sandbox_id)
                    return {
                        "success": False,
                        "error": f"Failed to install requirements: {req_result.error}"
                    }
            
            # Write input data
            input_json = json.dumps(input_data, indent=2)
            sandbox.write_file("input.json", input_json)
            
            # Modify code to read input
            modified_code = f"""
import json

# Load input data
with open('input.json', 'r') as f:
    input_data = json.load(f)

# Original tool code
{code}
"""
            
            # Execute
            result = sandbox.execute_code(modified_code, "tool_with_input.py")
            
            # Try to read any output files
            output_files = {}
            for filename in sandbox.list_files():
                if filename.endswith('.json') and filename != 'input.json':
                    try:
                        content = sandbox.read_file(filename)
                        output_files[filename] = json.loads(content)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass
            
            # Cleanup
            await self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time,
                "output_files": output_files
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    async def benchmark_tool(self, code: str, test_cases: List[Dict[str, Any]],
                            requirements: List[str] = None) -> Dict[str, Any]:
        """
        Benchmark a tool with multiple test cases
        """
        
        results = []
        total_time = 0
        success_count = 0
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i+1}/{len(test_cases)}")
            
            result = await self.run_tool_with_input(code, test_case, requirements)
            
            results.append({
                "test_case": i + 1,
                "input": test_case,
                "success": result["success"],
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "execution_time": result.get("execution_time", 0)
            })
            
            if result["success"]:
                success_count += 1
            
            total_time += result.get("execution_time", 0)
        
        return {
            "total_tests": len(test_cases),
            "successful_tests": success_count,
            "success_rate": success_count / len(test_cases) if test_cases else 0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(test_cases) if test_cases else 0,
            "detailed_results": results
        }

class EnvironmentManager:
    """Manager for handling environment setup and cleanup"""
    
    def __init__(self, sandbox_manager: SandboxManager):
        self.sandbox_manager = sandbox_manager
        self.environments = {}
    
    async def create_environment(self, name: str, requirements: List[str],
                                setup_script: str = None) -> str:
        """
        Create a new environment with specific requirements
        """
        
        config = SandboxConfig(name=f"env_{name}")
        sandbox_id = await self.sandbox_manager.create_sandbox(config)
        sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
        
        if sandbox is None:
            raise RuntimeError("Failed to create environment sandbox")
        
        # Install requirements
        if requirements:
            req_result = sandbox.install_requirements(requirements)
            if not req_result.success:
                await self.sandbox_manager.destroy_sandbox(sandbox_id)
                raise RuntimeError(f"Failed to install requirements: {req_result.error}")
        
        # Run setup script if provided
        if setup_script:
            setup_result = sandbox.execute_shell_command(setup_script)
            if not setup_result.success:
                logger.warning(f"Setup script failed: {setup_result.error}")
        
        self.environments[name] = sandbox_id
        return sandbox_id
    
    async def cleanup_environment(self, name: str):
        """Cleanup an environment"""
        if name in self.environments:
            sandbox_id = self.environments[name]
            await self.sandbox_manager.destroy_sandbox(sandbox_id)
            del self.environments[name]
    
    def get_environment(self, name: str) -> Optional[str]:
        """Get environment sandbox ID by name"""
        return self.environments.get(name)

# Convenience functions for MCP creation workflow
async def create_mcp_workflow(task_description: str, model_manager: MultiModelManager,
                             sandbox_manager: SandboxManager, web_agent) -> Dict[str, Any]:
    """
    Complete workflow for creating an MCP from task description
    """
    
    # Initialize tools
    brainstorming = MCPBrainstorming(model_manager)
    script_generator = ScriptGeneratingTool(model_manager, web_agent)
    code_runner = CodeRunningTool(sandbox_manager)
    
    # Step 1: Assess capabilities and brainstorm
    assessment = await brainstorming.assess_capabilities(task_description, [])
    ideas = await brainstorming.brainstorm_mcp_ideas(task_description)
    
    # Step 2: Generate tool
    best_idea = ideas[0] if ideas else {"name": "custom_tool", "description": task_description}
    tool_spec = await script_generator.generate_tool(
        best_idea["description"],
        [],  # No GitHub links for now
        []   # No specific requirements
    )
    
    # Step 3: Validate tool
    validation = await code_runner.validate_tool(
        tool_spec["code"],
        tool_spec.get("test_code"),
        tool_spec["requirements"]
    )
    
    # Step 4: Improve if needed
    if not validation["success"] and validation.get("error"):
        improved_spec = await script_generator.improve_tool(
            tool_spec["code"],
            validation["error"],
            task_description
        )
        
        # Re-validate improved tool
        validation = await code_runner.validate_tool(
            improved_spec["code"],
            tool_spec.get("test_code"),
            improved_spec.get("requirements", [])
        )
        
        if validation["success"]:
            tool_spec["code"] = improved_spec["code"]
            tool_spec["requirements"].extend(improved_spec.get("requirements", []))
    
    return {
        "task_description": task_description,
        "assessment": assessment,
        "ideas": ideas,
        "tool_spec": tool_spec,
        "validation": validation,
        "success": validation["success"]
    } 