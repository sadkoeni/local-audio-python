"""
Lightberry Tool Server

This server receives tool call JSON objects via LiveKit data channel and
executes the corresponding functions defined in local_tool_responses.py
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime

# Import the tool registry from local_tool_responses
import local_tool_responses
from local_tool_responses import TOOL_REGISTRY, get_available_tools

logger = logging.getLogger(__name__)


class LightberryToolServer:
    """
    Server that processes tool calls received via LiveKit data channel.
    
    This server:
    1. Receives JSON tool call objects
    2. Validates and formats the arguments
    3. Executes the corresponding tool function
    4. Returns the result via the data channel
    """
    
    def __init__(self, data_channel_name: Optional[str] = None):
        """
        Initialize the tool server.
        
        Args:
            data_channel_name: Name of the data channel to use (can be configured later)
        """
        self.data_channel_name = data_channel_name
        self.running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._response_callback: Optional[Callable] = None
        
        # Load all tools from local_tool_responses
        self._load_tools()
        
    def _load_tools(self):
        """Load all available tools from the registry."""
        tools = get_available_tools()
        logger.info(f"Loaded {len(tools)} tools: {list(tools.keys())}")
        
    def set_data_channel_name(self, name: str):
        """Set or update the data channel name."""
        self.data_channel_name = name
        logger.info(f"Data channel name set to: {name}")
        
    def set_response_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set the callback function to send responses back via data channel.
        
        Args:
            callback: Function that takes a response dict and sends it via data channel
        """
        self._response_callback = callback
        
    async def process_tool_call(self, message: str) -> Dict[str, Any]:
        """
        Process a tool call message received from the data channel.
        
        Args:
            message: JSON string containing the tool call
            
        Returns:
            Dictionary containing the result or error
        """
        try:
            # Parse the JSON message
            tool_call = json.loads(message)
            logger.debug(f"Received tool call: {tool_call}")
            
            # Validate the tool call structure
            if not isinstance(tool_call, dict):
                raise ValueError("Tool call must be a JSON object")
                
            # Extract tool name and arguments
            tool_name = tool_call.get("name")
            if not tool_name:
                raise ValueError("Tool call must include a 'name' field")
                
            # Format the arguments for the tool function
            formatted_args = self._format_tool_arguments(tool_call)
            
            # Execute the tool
            result = await self._execute_tool(tool_name, formatted_args)
            
            # Prepare response
            response = {
                "tool": tool_name,
                "status": "success",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tool call: {e}")
            return self._error_response("Invalid JSON format", str(e))
            
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            return self._error_response("Tool execution failed", str(e))
            
    def _format_tool_arguments(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the tool call arguments for the function.
        
        This is the central place to handle argument formatting and transformation.
        
        Args:
            tool_call: The raw tool call dictionary
            
        Returns:
            Formatted arguments ready to be passed to the tool function
        """
        # Remove the 'name' field as it's not an argument
        args = tool_call.copy()
        args.pop("name", None)
        
        # Remove any system fields that might be present
        system_fields = ["id", "timestamp", "source", "metadata"]
        for field in system_fields:
            args.pop(field, None)
            
        # Special handling for 'arguments' field if present
        # Some tool calls might wrap arguments in an 'arguments' object
        if "arguments" in args and isinstance(args["arguments"], dict):
            # Unwrap the arguments
            args = args["arguments"]
            
        # Convert any None values to proper defaults based on type hints
        # This ensures tools receive expected types
        
        logger.debug(f"Formatted arguments: {args}")
        return args
        
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool function with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            The result from the tool function
        """
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        tool_func = TOOL_REGISTRY[tool_name]
        
        # Check if the tool is async
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**args)
        else:
            # Run sync functions in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: tool_func(**args))
            
        return result
        
    def _error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "status": "error",
            "error": error_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def handle_data_message(self, message: str):
        """
        Handle a message received from the data channel.
        
        This method should be called by the LiveKit client when a message is received.
        """
        # Add message to queue for processing
        await self._message_queue.put(message)
        
    async def start(self):
        """Start the tool server."""
        self.running = True
        logger.info("Lightberry Tool Server started")
        
        # Start the message processing loop
        asyncio.create_task(self._process_message_loop())
        
    async def stop(self):
        """Stop the tool server."""
        self.running = False
        logger.info("Lightberry Tool Server stopped")
        
    async def _process_message_loop(self):
        """Process messages from the queue."""
        while self.running:
            try:
                # Wait for a message with timeout to allow checking running status
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                # Process the tool call
                response = await self.process_tool_call(message)
                
                # Send response if callback is set
                if self._response_callback:
                    await self._send_response(response)
                else:
                    logger.warning("No response callback set, result not sent back")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                
    async def _send_response(self, response: Dict[str, Any]):
        """Send a response back via the data channel."""
        try:
            if asyncio.iscoroutinefunction(self._response_callback):
                await self._response_callback(response)
            else:
                self._response_callback(response)
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            
    def get_tool_info(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available tools.
        
        Args:
            tool_name: Specific tool to get info for, or None for all tools
            
        Returns:
            Dictionary with tool information
        """
        if tool_name:
            if tool_name in TOOL_REGISTRY:
                tools = get_available_tools()
                return tools[tool_name]
            else:
                return {"error": f"Tool '{tool_name}' not found"}
        else:
            return get_available_tools()
            

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_server():
        """Test the tool server with example tool calls."""
        server = LightberryToolServer()
        
        # Set a simple response callback for testing
        def print_response(response):
            print(f"Response: {json.dumps(response, indent=2)}")
            
        server.set_response_callback(print_response)
        
        # Start the server
        await server.start()
        
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
                "expression": "42 * 2"
            },
            {
                "name": "template",
                "custom_arg1": "value1",
                "custom_arg2": 123,
                "nested": {"key": "value"}
            }
        ]
        
        for call in test_calls:
            print(f"\nTesting tool call: {call['name']}")
            response = await server.process_tool_call(json.dumps(call))
            print(f"Result: {json.dumps(response, indent=2)}")
            
        # Stop the server
        await server.stop()
        
    # Run the test
    asyncio.run(test_server())