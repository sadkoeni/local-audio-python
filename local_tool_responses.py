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


# ============== User Tool Definitions ==============
# Users can add their own tools below this line

# Template function as requested
@tool(name="template", description="Template tool function for demonstration")
def template_function(**kwargs) -> Dict[str, Any]:
    """
    Template function that accepts any keyword arguments.
    This demonstrates how to create a flexible tool that can handle various inputs.
    
    All arguments passed in the JSON tool call will be available as kwargs.
    """
    # Print the tool name and arguments as requested
    print(f"Tool called: template")
    print(f"Arguments: {kwargs}")
    
    return {
        "tool": "template",
        "received_arguments": kwargs,
        "argument_count": len(kwargs),
        "message": "This is a template function. Replace with your own implementation."
    }