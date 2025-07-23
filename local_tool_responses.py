"""
Local Tool Responses Module

This module allows users to define custom tool functions that can be executed
when tool calls are received via the LiveKit data channel.

To create a new tool:
1. Define a function with any name
2. Decorate it with @tool(name="tool_name")
3. The function will receive arguments from the JSON tool call

Example:
    @tool(name="get_weather")
    def fetch_weather(location: str, unit: str = "celsius") -> dict:
        # Your implementation here
        return {"temperature": 22, "unit": unit, "location": location}
"""

from typing import Any, Callable, Dict, Optional
import functools
import logging

logger = logging.getLogger(__name__)

# Registry to store all available tools
TOOL_REGISTRY: Dict[str, Callable] = {}


def tool(name: str, description: Optional[str] = None):
    """
    Decorator to register a function as a tool that can be called remotely.
    
    Args:
        name: The name that will be used to invoke this tool
        description: Optional description of what the tool does
        
    Example:
        @tool(name="calculator", description="Performs basic calculations")
        def calculate(operation: str, a: float, b: float) -> float:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Store metadata with the function
        func._tool_name = name
        func._tool_description = description or func.__doc__ or "No description available"
        
        # Register the function
        TOOL_REGISTRY[name] = func
        logger.info(f"Registered tool '{name}': {func._tool_description}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Executing tool '{name}' with args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                logger.debug(f"Tool '{name}' completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error executing tool '{name}': {e}")
                raise
                
        return wrapper
    return decorator


def get_available_tools() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all registered tools.
    
    Returns:
        Dictionary mapping tool names to their metadata
    """
    tools = {}
    for name, func in TOOL_REGISTRY.items():
        tools[name] = {
            "name": name,
            "description": getattr(func, '_tool_description', 'No description'),
            "function": func.__name__,
            "module": func.__module__
        }
    return tools


# ============== Example Tool Definitions ==============
# Users can add their own tools below this line

@tool(name="echo", description="Echoes back the input message")
def echo_message(message: str) -> Dict[str, str]:
    """Simple echo tool that returns the input message."""
    return {
        "original_message": message,
        "echo": message,
        "timestamp": str(time.time())
    }


@tool(name="system_info", description="Get basic system information")
def get_system_info() -> Dict[str, Any]:
    """Returns basic information about the system."""
    import platform
    import os
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "current_directory": os.getcwd()
    }


@tool(name="file_operations", description="Perform basic file operations")
def file_operations(operation: str, path: str, content: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform basic file operations.
    
    Args:
        operation: One of 'read', 'write', 'exists', 'list'
        path: File or directory path
        content: Content to write (only for 'write' operation)
    """
    import os
    
    result = {"operation": operation, "path": path, "success": False}
    
    try:
        if operation == "read":
            with open(path, 'r') as f:
                result["content"] = f.read()
                result["success"] = True
                
        elif operation == "write":
            if content is None:
                result["error"] = "Content required for write operation"
            else:
                with open(path, 'w') as f:
                    f.write(content)
                result["success"] = True
                result["bytes_written"] = len(content)
                
        elif operation == "exists":
            result["exists"] = os.path.exists(path)
            result["is_file"] = os.path.isfile(path) if os.path.exists(path) else False
            result["is_directory"] = os.path.isdir(path) if os.path.exists(path) else False
            result["success"] = True
            
        elif operation == "list":
            if os.path.isdir(path):
                result["files"] = os.listdir(path)
                result["success"] = True
            else:
                result["error"] = "Path is not a directory"
                
        else:
            result["error"] = f"Unknown operation: {operation}"
            
    except Exception as e:
        result["error"] = str(e)
        
    return result


@tool(name="calculate", description="Perform basic mathematical calculations")
def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    import ast
    import operator
    
    # Define safe operations
    safe_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return safe_ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            return safe_ops[type(node.op)](operand)
        else:
            raise ValueError(f"Unsafe operation: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


# Template function as requested
@tool(name="template", description="Template tool function for demonstration")
def template_function(**kwargs) -> Dict[str, Any]:
    """
    Template function that accepts any keyword arguments.
    This demonstrates how to create a flexible tool that can handle various inputs.
    
    All arguments passed in the JSON tool call will be available as kwargs.
    """
    return {
        "tool": "template",
        "received_arguments": kwargs,
        "argument_count": len(kwargs),
        "message": "This is a template function. Replace with your own implementation."
    }


# Import time for the echo tool
import time