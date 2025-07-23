#!/usr/bin/env python3
"""
Simple test script to demonstrate the tool system without audio dependencies.
"""

import asyncio
import json
import logging
from lightberry_tool_server import LightberryToolServer
from local_tool_responses import get_available_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_tool_server():
    """Test the tool server with various tool calls."""
    print("\n" + "="*60)
    print("LIGHTBERRY TOOL SYSTEM TEST")
    print("="*60)
    
    # Create server
    server = LightberryToolServer()
    await server.start()
    
    # List available tools
    tools = get_available_tools()
    print("\nAvailable tools:")
    for name, info in tools.items():
        print(f"  📦 {name}: {info['description']}")
    
    print("\n" + "-"*60)
    print("Testing tool calls...")
    print("-"*60)
    
    # Test tool calls
    test_calls = [
        {
            "name": "echo",
            "message": "Hello, Lightberry!"
        },
        {
            "name": "system_info"
        },
        {
            "name": "calculate",
            "expression": "2 + 2 * 3"
        },
        {
            "name": "file_operations",
            "operation": "exists",
            "path": "local_tool_responses.py"
        },
        {
            "name": "template",
            "custom_arg1": "test_value",
            "custom_arg2": 42,
            "nested_data": {"key": "value", "list": [1, 2, 3]}
        }
    ]
    
    for call in test_calls:
        print(f"\n🔧 Testing tool: {call['name']}")
        print(f"   Request: {json.dumps(call, indent=2)}")
        
        # Process the tool call
        response = await server.process_tool_call(json.dumps(call))
        
        if response['status'] == 'success':
            print(f"   ✅ Success!")
            print(f"   Response: {json.dumps(response['result'], indent=2)}")
        else:
            print(f"   ❌ Error: {response['error']}")
            print(f"   Details: {response['details']}")
    
    # Test error handling
    print("\n" + "-"*60)
    print("Testing error handling...")
    print("-"*60)
    
    error_tests = [
        {
            "name": "nonexistent_tool"
        },
        {
            # Missing name field
            "argument": "value"
        },
        "invalid json",
    ]
    
    for i, test in enumerate(error_tests):
        print(f"\n🔧 Error test {i+1}")
        if isinstance(test, str):
            print(f"   Request: {test}")
            response = await server.process_tool_call(test)
        else:
            print(f"   Request: {json.dumps(test, indent=2)}")
            response = await server.process_tool_call(json.dumps(test))
        
        print(f"   Expected error - Status: {response['status']}")
        print(f"   Error: {response.get('error', 'N/A')}")
        print(f"   Details: {response.get('details', 'N/A')}")
    
    # Stop server
    await server.stop()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_tool_server())