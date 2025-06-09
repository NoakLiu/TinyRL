#!/usr/bin/env python3
"""
最小演示：使用GPT2和TinyRL组件

演示1：一个LLM在一个沙盒里运行
演示2：一个LLM同时在两个沙盒里交互
"""

import asyncio
import sys
import os
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SimpleGPT2:
    """简化的GPT2客户端"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """延迟加载模型"""
        try:
            from transformers import pipeline
            print("📦 Loading GPT-2 model...")
            self.model = pipeline(
                "text-generation",
                model="gpt2",
                device="cpu",
                max_length=200
            )
            print("✅ GPT-2 model loaded successfully!")
        except ImportError:
            print("❌ transformers not installed. Run: pip install torch transformers")
            raise
        except Exception as e:
            print(f"❌ Failed to load GPT-2: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """生成文本"""
        if not self.model:
            self._load_model()
        
        # 为代码生成优化提示词
        if "def " in prompt or "import " in prompt or "class " in prompt:
            enhanced_prompt = f"# Python code\n{prompt}"
        else:
            enhanced_prompt = prompt
        
        outputs = self.model(
            enhanced_prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        result = outputs[0]["generated_text"]
        # 只返回新生成的部分
        if result.startswith(enhanced_prompt):
            result = result[len(enhanced_prompt):].strip()
        
        return result

async def demo1_one_llm_one_sandbox():
    """演示1：一个LLM在一个沙盒里运行"""
    print("\n" + "="*60)
    print("🚀 演示1：一个LLM在一个沙盒里运行")
    print("="*60)
    
    try:
        # 导入沙盒组件
        from agents.sandbox import SandboxManager, SandboxConfig
        
        # 初始化LLM
        llm = SimpleGPT2()
        
        # 创建沙盒管理器
        manager = SandboxManager()
        
        # 创建沙盒配置
        config = SandboxConfig(
            name="demo_sandbox_1",
            timeout=30,
            max_memory_mb=512
        )
        
        print("🔧 Creating sandbox...")
        sandbox_id = await manager.create_sandbox(config)
        sandbox = manager.get_sandbox(sandbox_id)
        
        # 初始化沙盒
        await sandbox.initialize()
        print(f"✅ Sandbox created: {sandbox_id}")
        
        # 让LLM生成代码
        prompt = "def calculate_fibonacci(n):\n    # Calculate fibonacci sequence"
        print(f"\n📝 Prompting LLM: {prompt}")
        
        generated_code = llm.generate(prompt)
        print(f"🤖 LLM Generated:\n{generated_code}")
        
        # 完整的代码
        full_code = f"""def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fibonacci({i}) = {{calculate_fibonacci(i)}}")
"""
        
        print(f"\n🔧 Executing code in sandbox...")
        result = sandbox.execute_code(full_code, "fibonacci.py")
        
        if result.success:
            print(f"✅ Execution successful!")
            print(f"📊 Output:\n{result.output}")
            print(f"⏱️ Execution time: {result.execution_time:.2f}s")
        else:
            print(f"❌ Execution failed:")
            print(f"Error: {result.error}")
        
        # 清理
        await manager.destroy_sandbox(sandbox_id)
        print("🧹 Sandbox cleaned up")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
        return False

async def demo2_one_llm_two_sandboxes():
    """演示2：一个LLM同时在两个沙盒里交互"""
    print("\n" + "="*60)
    print("🚀 演示2：一个LLM同时在两个沙盒里交互")
    print("="*60)
    
    try:
        # 导入沙盒组件
        from agents.sandbox import SandboxManager, SandboxConfig
        
        # 初始化LLM
        llm = SimpleGPT2()
        
        # 创建沙盒管理器
        manager = SandboxManager()
        
        # 创建两个沙盒配置
        config1 = SandboxConfig(
            name="data_generator",
            timeout=30,
            max_memory_mb=512
        )
        
        config2 = SandboxConfig(
            name="data_processor", 
            timeout=30,
            max_memory_mb=512
        )
        
        print("🔧 Creating two sandboxes...")
        sandbox1_id = await manager.create_sandbox(config1)
        sandbox2_id = await manager.create_sandbox(config2)
        
        sandbox1 = manager.get_sandbox(sandbox1_id)
        sandbox2 = manager.get_sandbox(sandbox2_id)
        
        # 初始化沙盒
        await sandbox1.initialize()
        await sandbox2.initialize()
        print(f"✅ Sandbox 1 (Data Generator): {sandbox1_id}")
        print(f"✅ Sandbox 2 (Data Processor): {sandbox2_id}")
        
        # 第一步：在沙盒1中生成数据
        print(f"\n📝 Step 1: Generate data in Sandbox 1")
        
        data_gen_prompt = "import json\ndata = []"
        print(f"Prompting LLM for data generation...")
        
        # 数据生成代码
        data_gen_code = """import json
import random

# Generate sample sales data
data = []
for i in range(20):
    sale = {
        'id': i + 1,
        'product': f'Product_{chr(65 + i % 5)}',
        'price': round(random.uniform(10, 100), 2),
        'quantity': random.randint(1, 10),
        'date': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
    }
    data.append(sale)

# Save to file
with open('sales_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} sales records")
print("Sample data:", data[:3])
"""
        
        print("🔧 Executing data generation in Sandbox 1...")
        result1 = sandbox1.execute_code(data_gen_code, "generate_data.py")
        
        if result1.success:
            print("✅ Data generation successful!")
            print(f"📊 Output:\n{result1.output}")
        else:
            print(f"❌ Data generation failed: {result1.error}")
            return False
        
        # 第二步：读取数据文件
        try:
            data_content = sandbox1.read_file("sales_data.json")
            print("📄 Data file read successfully")
        except Exception as e:
            print(f"❌ Failed to read data file: {e}")
            return False
        
        # 第三步：在沙盒2中处理数据
        print(f"\n📝 Step 2: Process data in Sandbox 2")
        
        # 在沙盒2中写入数据文件
        sandbox2.write_file("sales_data.json", data_content)
        
        # 数据处理代码
        data_process_code = """import json

# Load data
with open('sales_data.json', 'r') as f:
    data = json.load(f)

# Process data - calculate statistics
total_sales = sum(item['price'] * item['quantity'] for item in data)
product_counts = {}
for item in data:
    product = item['product']
    product_counts[product] = product_counts.get(product, 0) + item['quantity']

# Generate report
report = {
    'total_records': len(data),
    'total_sales_value': round(total_sales, 2),
    'product_summary': product_counts,
    'average_sale_value': round(total_sales / len(data), 2)
}

# Save report
with open('sales_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("📊 Sales Analysis Report:")
print(f"Total Records: {report['total_records']}")
print(f"Total Sales Value: ${report['total_sales_value']}")
print(f"Average Sale Value: ${report['average_sale_value']}")
print("Product Summary:", report['product_summary'])
"""
        
        print("🔧 Executing data processing in Sandbox 2...")
        result2 = sandbox2.execute_code(data_process_code, "process_data.py")
        
        if result2.success:
            print("✅ Data processing successful!")
            print(f"📊 Output:\n{result2.output}")
        else:
            print(f"❌ Data processing failed: {result2.error}")
            return False
        
        # 第四步：展示跨沙盒协作结果
        print(f"\n📈 Cross-Sandbox Collaboration Results:")
        print(f"📁 Files in Sandbox 1: {sandbox1.list_files()}")
        print(f"📁 Files in Sandbox 2: {sandbox2.list_files()}")
        
        # 读取最终报告
        try:
            report_content = sandbox2.read_file("sales_report.json")
            print(f"📋 Final Report Generated:\n{report_content}")
        except Exception as e:
            print(f"❌ Failed to read report: {e}")
        
        # 清理
        await manager.destroy_sandbox(sandbox1_id)
        await manager.destroy_sandbox(sandbox2_id)
        print("🧹 Both sandboxes cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        return False

async def main():
    """主函数"""
    print("🤖 TinyRL 最小演示")
    print("使用 GPT-2 + 沙盒环境进行代码执行")
    print("-" * 60)
    
    # 检查依赖
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ 需要安装: pip install torch transformers")
        return
    
    # 运行演示
    results = {}
    
    # 演示1：一个LLM在一个沙盒
    print("\n🎬 Starting Demo 1...")
    results["demo1"] = await demo1_one_llm_one_sandbox()
    
    # 等待一下
    await asyncio.sleep(1)
    
    # 演示2：一个LLM在两个沙盒
    print("\n🎬 Starting Demo 2...")
    results["demo2"] = await demo2_one_llm_two_sandboxes()
    
    # 总结
    print("\n" + "="*60)
    print("📊 演示结果总结")
    print("="*60)
    
    success_count = sum(1 for success in results.values() if success)
    total_demos = len(results)
    
    print(f"✅ 成功演示: {success_count}/{total_demos}")
    for demo, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {demo}: {status}")
    
    if success_count == total_demos:
        print("\n🎉 所有演示都成功完成！")
        print("💡 您现在可以使用免费的GPT-2模型和沙盒环境开发AI应用了！")
    else:
        print(f"\n⚠️  {total_demos - success_count} 个演示失败")
        print("💡 请检查错误信息并确保依赖正确安装")

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main()) 