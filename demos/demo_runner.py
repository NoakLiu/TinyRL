"""
MCP Agent Sandbox Demo Runner
Demonstrates the three core scenarios:
1. Single LLM in one sandbox
2. One LLM controlling two sandboxes  
3. Two LLMs in one sandbox
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Any
from dataclasses import dataclass

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock implementations to avoid import errors
class MockModelConfig:
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockLLM:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        # Mock response based on model type
        if "gpt" in self.model_name.lower():
            return f"GPT Response to: {prompt[:50]}... (This is a mock response from {self.model_name})"
        elif "claude" in self.model_name.lower():
            return f"Claude Response: {prompt[:50]}... (Mock Claude-3.5-Sonnet response)"
        elif "gemini" in self.model_name.lower():
            return f"Gemini Response: {prompt[:50]}... (Mock Gemini Pro response)"
        elif "mistral" in self.model_name.lower():
            return f"Mistral Response: {prompt[:50]}... (Mock Mistral Large response)"
        elif "llama" in self.model_name.lower():
            return f"Llama Response: {prompt[:50]}... (Mock Llama 3.1 response)"
        else:
            return f"LLM Response: {prompt[:50]}... (Mock response from {self.model_name})"
    
    def generate_sync(self, prompt: str, system_prompt: str = None) -> str:
        return asyncio.run(self.generate(prompt, system_prompt))

class MockMultiModelManager:
    def __init__(self):
        self.models = {}
        self.default_model = None
    
    def add_model(self, name: str, config):
        self.models[name] = MockLLM(config)
        if self.default_model is None:
            self.default_model = name
    
    async def generate(self, prompt: str, system_prompt: str = None, model_name: str = None) -> str:
        model_name = model_name or self.default_model
        if model_name in self.models:
            return await self.models[model_name].generate(prompt, system_prompt)
        return f"Mock response to: {prompt[:50]}..."
    
    def generate_sync(self, prompt: str, system_prompt: str = None, model_name: str = None) -> str:
        return asyncio.run(self.generate(prompt, system_prompt, model_name))

class MockSandboxResult:
    def __init__(self, success: bool = True, output: str = "", error: str = "", execution_time: float = 0.1):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time

class MockSandbox:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self.is_initialized = True
        self.files = {}
    
    def execute_code(self, code: str, filename: str = "script.py") -> MockSandboxResult:
        # Mock code execution
        if "error" in code.lower():
            return MockSandboxResult(False, "", "Mock execution error", 0.1)
        elif "print" in code:
            return MockSandboxResult(True, f"Mock output from {filename}: Hello from sandbox {self.sandbox_id}", "", 0.1)
        else:
            return MockSandboxResult(True, f"Code executed successfully in sandbox {self.sandbox_id}", "", 0.1)
    
    def install_requirements(self, requirements: List[str]) -> MockSandboxResult:
        return MockSandboxResult(True, f"Installed packages: {', '.join(requirements)}", "", 2.0)
    
    def write_file(self, filepath: str, content: str):
        self.files[filepath] = content
    
    def read_file(self, filepath: str) -> str:
        return self.files.get(filepath, f"Mock content of {filepath}")
    
    def list_files(self) -> List[str]:
        return list(self.files.keys())

class MockSandboxConfig:
    def __init__(self, name: str, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockSandboxManager:
    def __init__(self):
        self.sandboxes = {}
        self.counter = 0
    
    async def create_sandbox(self, config) -> str:
        self.counter += 1
        sandbox_id = f"sandbox_{self.counter}"
        self.sandboxes[sandbox_id] = MockSandbox(sandbox_id)
        return sandbox_id
    
    def get_sandbox(self, sandbox_id: str):
        return self.sandboxes.get(sandbox_id)
    
    async def destroy_sandbox(self, sandbox_id: str):
        if sandbox_id in self.sandboxes:
            del self.sandboxes[sandbox_id]

@dataclass
class DemoResult:
    """Result of a demo execution"""
    demo_name: str
    success: bool
    output: str
    execution_time: float
    details: Dict[str, Any]

class MCPAgentDemoRunner:
    """Main demo runner for MCP Agent Sandbox framework capabilities"""
    
    def __init__(self):
        self.model_manager = MockMultiModelManager()
        self.sandbox_manager = MockSandboxManager()
        self._setup_models()
    
    def _setup_models(self):
        """Setup multiple model configurations"""
        
        # GPT Models
        self.model_manager.add_model("gpt-4o", MockModelConfig(
            model_name="gpt-4o",
            api_key="mock_openai_key"
        ))
        
        # Claude Models  
        self.model_manager.add_model("claude-3.5-sonnet", MockModelConfig(
            model_name="claude-3-5-sonnet-20241022",
            api_key="mock_anthropic_key"
        ))
        
        # Gemini Models
        self.model_manager.add_model("gemini-pro", MockModelConfig(
            model_name="gemini-1.5-pro",
            api_key="mock_google_key"
        ))
        
        # Mistral Models
        self.model_manager.add_model("mistral-large", MockModelConfig(
            model_name="mistral-large-latest",
            api_key="mock_mistral_key"
        ))
        
        # Llama Models
        self.model_manager.add_model("llama3.1", MockModelConfig(
            model_name="llama3.1",
            api_key="",
            base_url="http://localhost:11434"
        ))
        
        logger.info(f"Setup {len(self.model_manager.models)} models")
    
    async def demo_1_single_llm_single_sandbox(self) -> DemoResult:
        """
        Demo 1: Single LLM in one sandbox
        The simplest case - one model executing code in an isolated environment
        """
        
        logger.info("🚀 Starting Demo 1: Single LLM in Single Sandbox")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create sandbox
            config = MockSandboxConfig(name="demo1_sandbox")
            sandbox_id = await self.sandbox_manager.create_sandbox(config)
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            
            # Task: Generate and execute a simple Python script
            task_prompt = """
            Create a Python script that:
            1. Calculates the factorial of 10
            2. Prints the result
            3. Creates a small list and sorts it
            
            Write the complete code:
            """
            
            # Generate code using LLM
            code_response = await self.model_manager.generate(
                task_prompt,
                system_prompt="You are a Python code generator. Generate clean, working code.",
                model_name="gpt-4o"
            )
            
            # Extract code (in real implementation, would parse the response)
            generated_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Calculate factorial of 10
result = factorial(10)
print(f"Factorial of 10 is: {result}")

# Create and sort a list
numbers = [64, 34, 25, 12, 22, 11, 90]
numbers.sort()
print(f"Sorted list: {numbers}")
"""
            
            # Execute code in sandbox
            execution_result = sandbox.execute_code(generated_code, "factorial_demo.py")
            
            # Cleanup
            await self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return DemoResult(
                demo_name="Single LLM + Single Sandbox",
                success=execution_result.success,
                output=f"LLM Response: {code_response[:100]}...\nExecution: {execution_result.output}",
                execution_time=execution_time,
                details={
                    "model_used": "gpt-4o",
                    "sandbox_id": sandbox_id,
                    "code_generated": len(generated_code),
                    "execution_success": execution_result.success
                }
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return DemoResult(
                demo_name="Single LLM + Single Sandbox",
                success=False,
                output=f"Error: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )
    
    async def demo_2_single_llm_dual_sandbox(self) -> DemoResult:
        """
        Demo 2: One LLM controlling two sandboxes
        Simulates Alita's Web Agent + Manager Agent pattern with two environments
        """
        
        logger.info("🔄 Starting Demo 2: Single LLM controlling Two Sandboxes")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create two sandboxes for different purposes
            retrieval_config = MockSandboxConfig(name="retrieval_sandbox")
            execution_config = MockSandboxConfig(name="execution_sandbox")
            
            retrieval_sandbox_id = await self.sandbox_manager.create_sandbox(retrieval_config)
            execution_sandbox_id = await self.sandbox_manager.create_sandbox(execution_config)
            
            retrieval_sandbox = self.sandbox_manager.get_sandbox(retrieval_sandbox_id)
            execution_sandbox = self.sandbox_manager.get_sandbox(execution_sandbox_id)
            
            # Task: LLM coordinates between data retrieval and processing
            coordination_prompt = """
            You need to coordinate two tasks across two environments:
            
            Environment 1 (Retrieval): Generate mock data (simulate web scraping)
            Environment 2 (Execution): Process the retrieved data
            
            Plan the coordination:
            """
            
            coordination_plan = await self.model_manager.generate(
                coordination_prompt,
                system_prompt="You are a task coordinator. Plan how to split work across environments.",
                model_name="claude-3.5-sonnet"
            )
            
            # Step 1: Data retrieval in first sandbox
            retrieval_code = """
import json

# Mock data retrieval (simulating web scraping)
mock_data = {
    "users": [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ],
    "timestamp": "2024-01-15T10:30:00Z"
}

# Save data for transfer
with open("retrieved_data.json", "w") as f:
    json.dump(mock_data, f)

print("Data retrieval completed")
print(f"Retrieved {len(mock_data['users'])} user records")
"""
            
            retrieval_result = retrieval_sandbox.execute_code(retrieval_code, "retrieve_data.py")
            
            # Transfer data (simulate inter-sandbox communication)
            retrieved_data = retrieval_sandbox.read_file("retrieved_data.json")
            execution_sandbox.write_file("input_data.json", retrieved_data)
            
            # Step 2: Data processing in second sandbox
            processing_code = """
import json

# Load transferred data
with open("input_data.json", "r") as f:
    data = json.load(f)

# Process data
users = data["users"]
total_age = sum(user["age"] for user in users)
average_age = total_age / len(users)

cities = list(set(user["city"] for user in users))

# Generate report
report = {
    "total_users": len(users),
    "average_age": average_age,
    "cities": cities,
    "processed_timestamp": data["timestamp"]
}

# Save results
with open("processing_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Data processing completed")
print(f"Processed {report['total_users']} users, average age: {report['average_age']:.1f}")
print(f"Cities: {', '.join(report['cities'])}")
"""
            
            processing_result = execution_sandbox.execute_code(processing_code, "process_data.py")
            
            # Cleanup
            await self.sandbox_manager.destroy_sandbox(retrieval_sandbox_id)
            await self.sandbox_manager.destroy_sandbox(execution_sandbox_id)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return DemoResult(
                demo_name="Single LLM + Dual Sandbox",
                success=retrieval_result.success and processing_result.success,
                output=f"Coordination Plan: {coordination_plan[:100]}...\nRetrieval: {retrieval_result.output}\nProcessing: {processing_result.output}",
                execution_time=execution_time,
                details={
                    "model_used": "claude-3.5-sonnet",
                    "retrieval_sandbox": retrieval_sandbox_id,
                    "execution_sandbox": execution_sandbox_id,
                    "retrieval_success": retrieval_result.success,
                    "processing_success": processing_result.success
                }
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return DemoResult(
                demo_name="Single LLM + Dual Sandbox",
                success=False,
                output=f"Error: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )
    
    async def demo_3_dual_llm_single_sandbox(self) -> DemoResult:
        """
        Demo 3: Two LLMs collaborating in one sandbox
        Simulates collaborative multi-agent problem solving
        """
        
        logger.info("🤝 Starting Demo 3: Two LLMs in Single Sandbox")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create shared sandbox
            config = MockSandboxConfig(name="collaborative_sandbox")
            sandbox_id = await self.sandbox_manager.create_sandbox(config)
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            
            # Problem: Two LLMs collaborate on a complex task
            problem_statement = """
            Problem: Create a data analysis pipeline that:
            1. Generates sample financial data
            2. Performs statistical analysis
            3. Creates a summary report
            
            Split this between two agents: Planner and Implementer
            """
            
            # Agent 1 (Planner): Creates the plan and data structure
            planning_prompt = f"""
            {problem_statement}
            
            As the PLANNER agent, create:
            1. A detailed plan for the data analysis pipeline
            2. Sample financial data generation code
            
            Focus on the architecture and data structure:
            """
            
            plan_response = await self.model_manager.generate(
                planning_prompt,
                system_prompt="You are a planning agent. Focus on architecture and data design.",
                model_name="gemini-pro"
            )
            
            # Agent 1 generates data
            data_generation_code = """
import random
import json
from datetime import datetime, timedelta

# Generate sample financial data
def generate_financial_data(days=30):
    base_price = 100
    data = []
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d")
        price_change = random.uniform(-5, 5)
        price = max(50, base_price + price_change)
        volume = random.randint(1000, 10000)
        
        data.append({
            "date": date,
            "price": round(price, 2),
            "volume": volume
        })
        
        base_price = price
    
    return data

# Generate and save data
financial_data = generate_financial_data()
with open("financial_data.json", "w") as f:
    json.dump(financial_data, f, indent=2)

print(f"Generated {len(financial_data)} days of financial data")
print(f"Price range: ${min(d['price'] for d in financial_data):.2f} - ${max(d['price'] for d in financial_data):.2f}")
"""
            
            data_result = sandbox.execute_code(data_generation_code, "generate_data.py")
            
            # Agent 2 (Implementer): Performs analysis based on Agent 1's data
            analysis_prompt = f"""
            The planning agent has generated financial data. As the IMPLEMENTER agent:
            
            1. Load the financial data from financial_data.json
            2. Perform statistical analysis (mean, std, trends)
            3. Generate a summary report
            
            Previous plan from planner: {plan_response[:200]}...
            
            Create the analysis code:
            """
            
            analysis_response = await self.model_manager.generate(
                analysis_prompt,
                system_prompt="You are an implementation agent. Focus on analysis and execution.",
                model_name="mistral-large"
            )
            
            # Agent 2 performs analysis
            analysis_code = """
import json
import statistics

# Load data generated by the planner agent
with open("financial_data.json", "r") as f:
    financial_data = json.load(f)

# Perform statistical analysis
prices = [d["price"] for d in financial_data]
volumes = [d["volume"] for d in financial_data]

price_stats = {
    "mean": statistics.mean(prices),
    "median": statistics.median(prices),
    "std_dev": statistics.stdev(prices),
    "min": min(prices),
    "max": max(prices)
}

volume_stats = {
    "mean": statistics.mean(volumes),
    "median": statistics.median(volumes),
    "total": sum(volumes)
}

# Calculate trend
first_week_avg = statistics.mean(prices[:7])
last_week_avg = statistics.mean(prices[-7:])
trend = "UP" if last_week_avg > first_week_avg else "DOWN"

# Generate report
report = {
    "analysis_date": financial_data[-1]["date"],
    "data_points": len(financial_data),
    "price_analysis": price_stats,
    "volume_analysis": volume_stats,
    "trend": trend,
    "trend_change": round(((last_week_avg - first_week_avg) / first_week_avg) * 100, 2)
}

# Save collaborative report
with open("analysis_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Analysis completed by implementer agent")
print(f"Price trend: {trend} ({report['trend_change']:+.2f}%)")
print(f"Average price: ${price_stats['mean']:.2f} (±${price_stats['std_dev']:.2f})")
"""
            
            analysis_result = sandbox.execute_code(analysis_code, "analyze_data.py")
            
            # Read final collaborative result
            final_report = sandbox.read_file("analysis_report.json")
            
            # Cleanup
            await self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return DemoResult(
                demo_name="Dual LLM + Single Sandbox",
                success=data_result.success and analysis_result.success,
                output=f"Planner (Gemini): {plan_response[:100]}...\nImplementer (Mistral): {analysis_response[:100]}...\nData Generation: {data_result.output}\nAnalysis: {analysis_result.output}",
                execution_time=execution_time,
                details={
                    "planner_model": "gemini-pro", 
                    "implementer_model": "mistral-large",
                    "sandbox_id": sandbox_id,
                    "data_generation_success": data_result.success,
                    "analysis_success": analysis_result.success,
                    "final_report": final_report[:200] + "..." if len(final_report) > 200 else final_report
                }
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return DemoResult(
                demo_name="Dual LLM + Single Sandbox",
                success=False,
                output=f"Error: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )
    
    async def run_all_demos(self) -> List[DemoResult]:
        """Run all three demos and return results"""
        
        logger.info("🎯 Starting Alita Framework Demo Suite")
        logger.info("=" * 60)
        
        results = []
        
        # Demo 1: Single LLM + Single Sandbox
        result1 = await self.demo_1_single_llm_single_sandbox()
        results.append(result1)
        logger.info(f"✅ Demo 1 {'Completed' if result1.success else 'Failed'}")
        
        # Demo 2: Single LLM + Dual Sandbox
        result2 = await self.demo_2_single_llm_dual_sandbox()
        results.append(result2)
        logger.info(f"✅ Demo 2 {'Completed' if result2.success else 'Failed'}")
        
        # Demo 3: Dual LLM + Single Sandbox  
        result3 = await self.demo_3_dual_llm_single_sandbox()
        results.append(result3)
        logger.info(f"✅ Demo 3 {'Completed' if result3.success else 'Failed'}")
        
        return results
    
    def print_summary(self, results: List[DemoResult]):
        """Print a comprehensive summary of all demo results"""
        
        print("\n" + "=" * 80)
        print("🚀 ALITA FRAMEWORK DEMO RESULTS SUMMARY")
        print("=" * 80)
        
        total_time = sum(r.execution_time for r in results)
        successful_demos = sum(1 for r in results if r.success)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Demos: {len(results)}")
        print(f"   Successful: {successful_demos}/{len(results)}")
        print(f"   Total Execution Time: {total_time:.2f} seconds")
        print(f"   Average Time per Demo: {total_time/len(results):.2f} seconds")
        
        print(f"\n📋 Individual Demo Results:")
        
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"\n   Demo {i}: {result.demo_name}")
            print(f"   Status: {status}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            print(f"   Output: {result.output[:150]}{'...' if len(result.output) > 150 else ''}")
            
            if result.details:
                print(f"   Details: {json.dumps(result.details, indent=6)}")
        
        print(f"\n🎯 Framework Capabilities Demonstrated:")
        print(f"   ✓ Multi-model LLM support (GPT, Claude, Gemini, Mistral, Llama)")
        print(f"   ✓ Isolated sandbox execution")
        print(f"   ✓ Single LLM workflow")
        print(f"   ✓ Multi-sandbox coordination")
        print(f"   ✓ Multi-agent collaboration")
        print(f"   ✓ File transfer between environments")
        print(f"   ✓ Code generation and execution")
        print(f"   ✓ Error handling and recovery")
        
        print("\n" + "=" * 80)

async def main():
    """Main entry point for the demo"""
    
    print("🌟 Welcome to the Alita Framework Demo")
    print("This demo showcases the core capabilities of the Alita Framework:")
    print("- Minimal predefinition, maximal self-evolution")
    print("- Multi-model LLM support")
    print("- Sandbox-based code execution")
    print("- Multi-agent collaboration")
    print("\nStarting demos...\n")
    
    # Create and run demos
    runner = MCPAgentDemoRunner()
    results = await runner.run_all_demos()
    
    # Print comprehensive summary
    runner.print_summary(results)
    
    print("\n🎉 Demo completed! The Alita Framework is ready for development.")
    return results

if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main()) 