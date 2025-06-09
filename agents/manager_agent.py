"""
Manager Agent - Central Coordinator for MCP Agent Sandbox Framework
Implements minimal predefinition and maximal self-evolution principles
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from models.llm_interface import MultiModelManager
from agents.sandbox import SandboxManager, SandboxConfig, SandboxResult
from .web_agent import WebAgent
from .mcp_tools import MCPBrainstorming, ScriptGeneratingTool, CodeRunningTool

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a task to be executed"""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    subtasks: Optional[List['Task']] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []

@dataclass
class MCPSpec:
    """Specification for a Model Context Protocol"""
    name: str
    description: str
    requirements: List[str]
    code: str
    test_code: Optional[str] = None

class ManagerAgent:
    """Central coordinator implementing Alita's minimal predefinition philosophy"""
    
    def __init__(self, model_manager: MultiModelManager, 
                 sandbox_manager: SandboxManager,
                 web_agent: WebAgent):
        self.model_manager = model_manager
        self.sandbox_manager = sandbox_manager
        self.web_agent = web_agent
        
        # Core tools (minimal predefinition)
        self.mcp_brainstorming = MCPBrainstorming(model_manager)
        self.script_generator = ScriptGeneratingTool(model_manager, web_agent)
        self.code_runner = CodeRunningTool(sandbox_manager)
        
        # MCP registry for self-evolution
        self.mcp_box: Dict[str, MCPSpec] = {}
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        
    async def process_task(self, task_description: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for processing tasks using the CodeReAct loop
        Following Alita's architecture: Analysis -> MCP Brainstorming -> Tool Generation -> Execution
        """
        if task_id is None:
            task_id = f"task_{len(self.active_tasks)}"
        
        task = Task(task_id=task_id, description=task_description)
        self.active_tasks[task_id] = task
        
        try:
            logger.info(f"Starting task {task_id}: {task_description}")
            
            # Step 1: Initial task analysis
            analysis_result = await self._analyze_task(task_description)
            
            # Step 2: MCP Brainstorming - identify capability gaps
            brainstorm_result = await self.mcp_brainstorming.assess_capabilities(
                task_description, self._get_current_capabilities()
            )
            
            # Step 3: CodeReAct Loop - iterative problem solving
            final_result = await self._code_react_loop(
                task_description, analysis_result, brainstorm_result
            )
            
            task.status = TaskStatus.COMPLETED
            task.result = final_result
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": final_result,
                "mcps_created": list(self.mcp_box.keys())
            }
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze task and break it down into components"""
        
        analysis_prompt = f"""
        Analyze the following task and provide a structured breakdown:
        
        Task: {task_description}
        
        Please provide:
        1. Task type (research, computation, data processing, etc.)
        2. Required capabilities
        3. Potential subtasks
        4. Expected challenges
        5. Success criteria
        
        Format your response as JSON with keys: task_type, capabilities, subtasks, challenges, success_criteria
        """
        
        response = await self.model_manager.generate(
            analysis_prompt,
            system_prompt="You are an expert task analyst. Provide structured analysis in JSON format."
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "task_type": "general",
                "capabilities": ["basic_reasoning"],
                "subtasks": [task_description],
                "challenges": ["unknown"],
                "success_criteria": ["task_completion"]
            }
    
    async def _code_react_loop(self, task_description: str, 
                              analysis: Dict[str, Any], 
                              brainstorm: Dict[str, Any]) -> Any:
        """
        Implement the CodeReAct loop - iterative reasoning and action
        """
        max_iterations = 5
        iteration = 0
        context = {
            "task": task_description,
            "analysis": analysis,
            "brainstorm": brainstorm,
            "actions_taken": [],
            "intermediate_results": []
        }
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"CodeReAct iteration {iteration}")
            
            # Reasoning phase
            reasoning_result = await self._reasoning_phase(context)
            
            # Action phase
            action_result = await self._action_phase(reasoning_result)
            
            # Update context
            context["actions_taken"].append(action_result)
            context["intermediate_results"].append(action_result.get("result"))
            
            # Check if task is complete
            if action_result.get("task_complete", False):
                return action_result.get("final_result")
            
            # Check if we need new tools
            if action_result.get("need_new_tools", False):
                mcp_created = await self._create_mcp(action_result.get("tool_requirements", {}))
                context["new_mcps"] = context.get("new_mcps", []) + [mcp_created]
        
        # If we reach max iterations, return the best result we have
        return context["intermediate_results"][-1] if context["intermediate_results"] else None
    
    async def _reasoning_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reasoning phase of CodeReAct loop"""
        
        reasoning_prompt = f"""
        Current context:
        Task: {context['task']}
        Analysis: {json.dumps(context['analysis'], indent=2)}
        Previous actions: {json.dumps(context['actions_taken'], indent=2)}
        
        Based on the context, determine the next action to take.
        
        Available actions:
        1. web_search - Search for information online
        2. execute_code - Run code in sandbox
        3. create_tool - Generate new MCP tool
        4. direct_solve - Attempt direct solution
        
        Respond with JSON containing:
        - action: one of the available actions
        - reasoning: why this action is needed
        - parameters: specific parameters for the action
        - expected_outcome: what you expect to achieve
        """
        
        response = await self.model_manager.generate(
            reasoning_prompt,
            system_prompt="You are a reasoning agent. Think step by step and choose the best next action."
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "action": "direct_solve",
                "reasoning": "Fallback to direct solving",
                "parameters": {"task": context["task"]},
                "expected_outcome": "Direct task completion"
            }
    
    async def _action_phase(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action phase of CodeReAct loop"""
        
        action = reasoning_result.get("action", "direct_solve")
        parameters = reasoning_result.get("parameters", {})
        
        if action == "web_search":
            return await self._handle_web_search(parameters)
        elif action == "execute_code":
            return await self._handle_code_execution(parameters)
        elif action == "create_tool":
            return await self._handle_tool_creation(parameters)
        elif action == "direct_solve":
            return await self._handle_direct_solve(parameters)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "task_complete": False
            }
    
    async def _handle_web_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search action"""
        try:
            query = parameters.get("query", "")
            search_results = await self.web_agent.search(query)
            
            return {
                "success": True,
                "action": "web_search",
                "result": search_results,
                "task_complete": False,
                "need_new_tools": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_complete": False
            }
    
    async def _handle_code_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code execution action"""
        try:
            code = parameters.get("code", "")
            requirements = parameters.get("requirements", [])
            
            # Create sandbox for execution
            config = SandboxConfig(name="code_execution")
            sandbox_id = await self.sandbox_manager.create_sandbox(config)
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            
            if sandbox is None:
                return {
                    "success": False,
                    "error": "Failed to create sandbox",
                    "task_complete": False
                }
            
            # Install requirements if needed
            if requirements:
                req_result = sandbox.install_requirements(requirements)
                if not req_result.success:
                    return {
                        "success": False,
                        "error": f"Failed to install requirements: {req_result.error}",
                        "task_complete": False
                    }
            
            # Execute code
            result = sandbox.execute_code(code)
            
            return {
                "success": result.success,
                "action": "execute_code",
                "result": result.output,
                "error": result.error if not result.success else None,
                "task_complete": parameters.get("is_final", False),
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_complete": False
            }
    
    async def _handle_tool_creation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool creation action"""
        try:
            tool_spec = parameters.get("tool_spec", {})
            mcp_created = await self._create_mcp(tool_spec)
            
            return {
                "success": True,
                "action": "create_tool",
                "result": f"Created MCP: {mcp_created['name']}",
                "mcp_created": mcp_created,
                "task_complete": False,
                "need_new_tools": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_complete": False
            }
    
    async def _handle_direct_solve(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct problem solving"""
        try:
            task = parameters.get("task", "")
            
            solve_prompt = f"""
            Solve the following task directly:
            {task}
            
            Provide a complete solution. If you need to write code, include it.
            If you need to search for information, describe what you would search for.
            """
            
            solution = await self.model_manager.generate(
                solve_prompt,
                system_prompt="You are a problem solver. Provide complete, actionable solutions."
            )
            
            return {
                "success": True,
                "action": "direct_solve",
                "result": solution,
                "final_result": solution,
                "task_complete": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_complete": False
            }
    
    async def _create_mcp(self, tool_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new MCP based on requirements"""
        
        # Use ScriptGeneratingTool to create the MCP
        mcp_spec = await self.script_generator.generate_tool(
            tool_requirements.get("description", ""),
            tool_requirements.get("github_links", []),
            tool_requirements.get("requirements", [])
        )
        
        # Test the generated MCP
        test_result = await self.code_runner.validate_tool(mcp_spec["code"])
        
        if test_result["success"]:
            # Store in MCP box
            mcp_name = tool_requirements.get("name", f"mcp_{len(self.mcp_box)}")
            self.mcp_box[mcp_name] = MCPSpec(
                name=mcp_name,
                description=mcp_spec["description"],
                requirements=mcp_spec["requirements"],
                code=mcp_spec["code"],
                test_code=mcp_spec.get("test_code")
            )
            
            logger.info(f"Created and stored MCP: {mcp_name}")
            return {"name": mcp_name, "success": True}
        else:
            raise Exception(f"MCP validation failed: {test_result['error']}")
    
    def _get_current_capabilities(self) -> List[str]:
        """Get list of current capabilities including MCPs"""
        base_capabilities = [
            "web_search",
            "code_execution", 
            "text_generation",
            "task_analysis",
            "mcp_creation"
        ]
        
        mcp_capabilities = [f"mcp_{name}" for name in self.mcp_box.keys()]
        
        return base_capabilities + mcp_capabilities
    
    def get_mcp_box_status(self) -> Dict[str, Any]:
        """Get status of the MCP box"""
        return {
            "total_mcps": len(self.mcp_box),
            "mcp_names": list(self.mcp_box.keys()),
            "capabilities": self._get_current_capabilities()
        }
    
    def export_mcp(self, mcp_name: str) -> Optional[Dict[str, Any]]:
        """Export an MCP for reuse"""
        if mcp_name in self.mcp_box:
            mcp = self.mcp_box[mcp_name]
            return {
                "name": mcp.name,
                "description": mcp.description,
                "requirements": mcp.requirements,
                "code": mcp.code,
                "test_code": mcp.test_code
            }
        return None
    
    def import_mcp(self, mcp_data: Dict[str, Any]):
        """Import an MCP into the MCP box"""
        mcp_spec = MCPSpec(
            name=mcp_data["name"],
            description=mcp_data["description"],
            requirements=mcp_data["requirements"],
            code=mcp_data["code"],
            test_code=mcp_data.get("test_code")
        )
        
        self.mcp_box[mcp_spec.name] = mcp_spec
        logger.info(f"Imported MCP: {mcp_spec.name}")

# Convenience function to create a fully configured manager agent
async def create_manager_agent(model_manager: Optional[MultiModelManager] = None) -> ManagerAgent:
    """Create a fully configured Manager Agent"""
    
    if model_manager is None:
        from models.llm_interface import create_model_manager_from_env
        model_manager = create_model_manager_from_env()
    
    # Create sandbox manager
    sandbox_manager = SandboxManager()
    
    # Create web agent
    web_agent = WebAgent(model_manager)
    
    # Create manager agent
    manager = ManagerAgent(model_manager, sandbox_manager, web_agent)
    
    return manager 