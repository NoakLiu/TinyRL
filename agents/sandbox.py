"""
Sandbox Execution Environment for Alita Framework
Supports isolated code execution with virtual environment management
"""

import os
import subprocess
import tempfile
import uuid
import shutil
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import concurrent.futures

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for sandbox environment"""
    name: str
    python_version: str = "3.9"
    timeout: int = 60
    max_memory_mb: int = 1024
    allow_network: bool = True
    working_dir: Optional[str] = None
    env_vars: Dict[str, str] = None

class SandboxResult:
    """Result of sandbox execution"""
    def __init__(self, success: bool, output: str, error: str = "", 
                 execution_time: float = 0.0, files_created: List[str] = None):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.files_created = files_created or []

class Sandbox:
    """Isolated execution environment"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.sandbox_id = str(uuid.uuid4())[:8]
        self.base_dir = None
        self.venv_path = None
        self.is_initialized = False
        self._lock = threading.Lock()
        
    async def initialize(self):
        """Initialize the sandbox environment"""
        with self._lock:
            if self.is_initialized:
                return
                
            try:
                # Create sandbox directory
                self.base_dir = Path(tempfile.mkdtemp(prefix=f"alita_sandbox_{self.sandbox_id}_"))
                self.working_dir = self.config.working_dir or str(self.base_dir / "workspace")
                os.makedirs(self.working_dir, exist_ok=True)
                
                # Create virtual environment
                self.venv_path = self.base_dir / "venv"
                subprocess.run([
                    "python", "-m", "venv", str(self.venv_path)
                ], check=True, capture_output=True)
                
                # Get python executable path
                if os.name == 'nt':  # Windows
                    self.python_exe = self.venv_path / "Scripts" / "python.exe"
                    self.pip_exe = self.venv_path / "Scripts" / "pip.exe"
                else:  # Unix-like
                    self.python_exe = self.venv_path / "bin" / "python"
                    self.pip_exe = self.venv_path / "bin" / "pip"
                
                # Upgrade pip
                subprocess.run([
                    str(self.pip_exe), "install", "--upgrade", "pip"
                ], check=True, capture_output=True)
                
                self.is_initialized = True
                logger.info(f"Sandbox {self.sandbox_id} initialized at {self.base_dir}")
                
            except Exception as e:
                logger.error(f"Failed to initialize sandbox {self.sandbox_id}: {e}")
                if self.base_dir and self.base_dir.exists():
                    shutil.rmtree(self.base_dir, ignore_errors=True)
                raise
    
    def install_requirements(self, requirements: List[str]) -> SandboxResult:
        """Install Python packages in the sandbox"""
        if not self.is_initialized:
            return SandboxResult(False, "", "Sandbox not initialized")
        
        if not requirements:
            return SandboxResult(True, "No requirements to install")
        
        try:
            # Create requirements.txt
            req_file = self.base_dir / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write('\n'.join(requirements))
            
            # Install requirements
            start_time = time.time()
            result = subprocess.run([
                str(self.pip_exe), "install", "-r", str(req_file)
            ], capture_output=True, text=True, timeout=self.config.timeout)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return SandboxResult(True, result.stdout, result.stderr, execution_time)
            else:
                return SandboxResult(False, result.stdout, result.stderr, execution_time)
                
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", "Installation timeout")
        except Exception as e:
            return SandboxResult(False, "", str(e))
    
    def execute_code(self, code: str, filename: str = "script.py") -> SandboxResult:
        """Execute Python code in the sandbox"""
        if not self.is_initialized:
            return SandboxResult(False, "", "Sandbox not initialized")
        
        try:
            # Write code to file
            script_path = Path(self.working_dir) / filename
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Prepare environment variables
            env = os.environ.copy()
            if self.config.env_vars:
                env.update(self.config.env_vars)
            
            # Execute code
            start_time = time.time()
            result = subprocess.run([
                str(self.python_exe), str(script_path)
            ], capture_output=True, text=True, timeout=self.config.timeout,
            cwd=self.working_dir, env=env)
            
            execution_time = time.time() - start_time
            
            # List files created
            files_created = []
            for file_path in Path(self.working_dir).iterdir():
                if file_path.name != filename:
                    files_created.append(str(file_path))
            
            if result.returncode == 0:
                return SandboxResult(True, result.stdout, result.stderr, 
                                   execution_time, files_created)
            else:
                return SandboxResult(False, result.stdout, result.stderr, 
                                   execution_time, files_created)
                
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", "Execution timeout")
        except Exception as e:
            return SandboxResult(False, "", str(e))
    
    def execute_shell_command(self, command: str) -> SandboxResult:
        """Execute shell command in the sandbox"""
        if not self.is_initialized:
            return SandboxResult(False, "", "Sandbox not initialized")
        
        try:
            # Prepare environment variables
            env = os.environ.copy()
            if self.config.env_vars:
                env.update(self.config.env_vars)
            
            # Execute command
            start_time = time.time()
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self.config.timeout, cwd=self.working_dir, env=env
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return SandboxResult(True, result.stdout, result.stderr, execution_time)
            else:
                return SandboxResult(False, result.stdout, result.stderr, execution_time)
                
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", "Command timeout")
        except Exception as e:
            return SandboxResult(False, "", str(e))
    
    def read_file(self, filepath: str) -> str:
        """Read file from sandbox"""
        full_path = Path(self.working_dir) / filepath
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_file(self, filepath: str, content: str):
        """Write file to sandbox"""
        full_path = Path(self.working_dir) / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def list_files(self) -> List[str]:
        """List all files in sandbox workspace"""
        files = []
        for root, dirs, filenames in os.walk(self.working_dir):
            for filename in filenames:
                rel_path = os.path.relpath(
                    os.path.join(root, filename), 
                    self.working_dir
                )
                files.append(rel_path)
        return files
    
    def cleanup(self):
        """Cleanup sandbox resources"""
        if self.base_dir and self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
                logger.info(f"Sandbox {self.sandbox_id} cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {self.sandbox_id}: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

class SandboxManager:
    """Manager for multiple sandbox environments"""
    
    def __init__(self, max_concurrent_sandboxes: int = 5):
        self.sandboxes: Dict[str, Sandbox] = {}
        self.max_concurrent = max_concurrent_sandboxes
        self._lock = threading.Lock()
    
    async def create_sandbox(self, config: SandboxConfig) -> str:
        """Create and initialize a new sandbox"""
        with self._lock:
            if len(self.sandboxes) >= self.max_concurrent:
                # Cleanup oldest sandbox
                oldest_id = next(iter(self.sandboxes))
                await self.destroy_sandbox(oldest_id)
            
            sandbox = Sandbox(config)
            await sandbox.initialize()
            
            self.sandboxes[sandbox.sandbox_id] = sandbox
            return sandbox.sandbox_id
    
    def get_sandbox(self, sandbox_id: str) -> Optional[Sandbox]:
        """Get sandbox by ID"""
        return self.sandboxes.get(sandbox_id)
    
    async def destroy_sandbox(self, sandbox_id: str):
        """Destroy sandbox and cleanup resources"""
        with self._lock:
            if sandbox_id in self.sandboxes:
                sandbox = self.sandboxes[sandbox_id]
                sandbox.cleanup()
                del self.sandboxes[sandbox_id]
    
    def list_sandboxes(self) -> List[str]:
        """List all active sandbox IDs"""
        return list(self.sandboxes.keys())
    
    async def cleanup_all(self):
        """Cleanup all sandboxes"""
        sandbox_ids = list(self.sandboxes.keys())
        for sandbox_id in sandbox_ids:
            await self.destroy_sandbox(sandbox_id)

class MultiSandboxRunner:
    """Runner for executing code across multiple sandboxes"""
    
    def __init__(self, manager: SandboxManager):
        self.manager = manager
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    async def run_parallel(self, tasks: List[Tuple[str, str, str]]) -> Dict[str, SandboxResult]:
        """
        Run multiple tasks in parallel across different sandboxes
        tasks: List of (sandbox_id, code, filename) tuples
        """
        results = {}
        
        def execute_task(task):
            sandbox_id, code, filename = task
            sandbox = self.manager.get_sandbox(sandbox_id)
            if sandbox:
                return sandbox_id, sandbox.execute_code(code, filename)
            else:
                return sandbox_id, SandboxResult(False, "", "Sandbox not found")
        
        # Submit all tasks
        futures = []
        for task in tasks:
            future = self.executor.submit(execute_task, task)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            sandbox_id, result = future.result()
            results[sandbox_id] = result
        
        return results
    
    def run_sequential(self, sandbox_id: str, codes: List[Tuple[str, str]]) -> List[SandboxResult]:
        """
        Run multiple code snippets sequentially in the same sandbox
        codes: List of (code, filename) tuples
        """
        sandbox = self.manager.get_sandbox(sandbox_id)
        if not sandbox:
            return [SandboxResult(False, "", "Sandbox not found")] * len(codes)
        
        results = []
        for code, filename in codes:
            result = sandbox.execute_code(code, filename)
            results.append(result)
        
        return results

# Convenience functions
async def create_quick_sandbox(name: str = "quick") -> Tuple[SandboxManager, str]:
    """Create a quick sandbox for testing"""
    manager = SandboxManager()
    config = SandboxConfig(name=name)
    sandbox_id = await manager.create_sandbox(config)
    return manager, sandbox_id

async def execute_in_new_sandbox(code: str, requirements: List[str] = None) -> SandboxResult:
    """Execute code in a new sandbox (auto-cleanup)"""
    manager = SandboxManager()
    config = SandboxConfig(name="temp")
    
    try:
        sandbox_id = await manager.create_sandbox(config)
        sandbox = manager.get_sandbox(sandbox_id)
        
        # Install requirements if needed
        if requirements:
            req_result = sandbox.install_requirements(requirements)
            if not req_result.success:
                return req_result
        
        # Execute code
        result = sandbox.execute_code(code)
        return result
        
    finally:
        await manager.cleanup_all() 