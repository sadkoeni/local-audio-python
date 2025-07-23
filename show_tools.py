#!/usr/bin/env python3
"""
Simple script to show available tools without requiring full dependencies.
"""

from local_tool_responses import get_available_tools

print("\n" + "="*60)
print("AVAILABLE TOOLS IN LIGHTBERRY SYSTEM")
print("="*60)

tools = get_available_tools()

for name, info in tools.items():
    print(f"\n📦 Tool: {name}")
    print(f"   Description: {info['description']}")
    print(f"   Function: {info['function']}")
    print(f"   Module: {info['module']}")

print("\n" + "="*60)
print(f"Total tools available: {len(tools)}")
print("="*60)

print("\nTo add more tools, edit 'local_tool_responses.py' and add functions")
print("decorated with @tool(name='your_tool_name')\n")