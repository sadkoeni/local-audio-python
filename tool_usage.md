# Lightberry Tool System Documentation

## Overview

The Lightberry Tool System allows you to execute functions on your local device through tool calls received via LiveKit's data channel. This enables remote agents or applications to interact with your local environment in a controlled and secure manner.

## Architecture

The system consists of three main components:

1. **Local Tool Responses** (`local_tool_responses.py`) - Defines the tools/functions that can be executed
2. **Lightberry Tool Server** (`lightberry_tool_server.py`) - Processes tool calls and executes functions
3. **Audio Stream with Tools** (`stream_audio_with_tools.py`) - Integrates tool execution with audio streaming

## Quick Start

### Running the System

```bash
# Basic usage
python lightberry_coordinator.py

# With custom participant name
python lightberry_coordinator.py --name "my-device"

# Using a specific audio device
python lightberry_coordinator.py --device 2

# With custom data channel name
python lightberry_coordinator.py --data-channel "my-tools"

# Enable debug logging
python lightberry_coordinator.py --log-level DEBUG
```

### Available Commands

```bash
# List all available tools
python lightberry_coordinator.py --list-tools

# Test the tool server locally
python lightberry_coordinator.py --test-tools

# List audio devices
python list_devices.py
```

## Tool Call JSON Format

Tool calls should be sent as JSON objects via the configured data channel. The format is:

```json
{
  "name": "tool_name",
  "argument1": "value1",
  "argument2": "value2",
  ...
}
```

### Required Fields

- `name` (string): The name of the tool to execute

### Optional System Fields

These fields are automatically stripped and not passed to the tool function:
- `id` - Request ID for tracking
- `timestamp` - When the request was created
- `source` - Origin of the request
- `metadata` - Additional metadata

### Arguments Format

All other fields in the JSON object are passed as keyword arguments to the tool function. If your tool call uses an `arguments` wrapper, it will be automatically unwrapped:

```json
{
  "name": "echo",
  "arguments": {
    "message": "Hello, World!"
  }
}
```

Is equivalent to:

```json
{
  "name": "echo",
  "message": "Hello, World!"
}
```

## Response Format

Tool execution results are sent back via the data channel in the following format:

### Success Response

```json
{
  "tool": "tool_name",
  "status": "success",
  "result": {
    // Tool-specific result data
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Error Response

```json
{
  "status": "error",
  "error": "Error type",
  "details": "Detailed error message",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

## Creating Custom Tools

To create your own tools, edit the `local_tool_responses.py` file and add functions decorated with `@tool`:

### Basic Tool Example

```python
@tool(name="my_tool", description="Does something useful")
def my_function(param1: str, param2: int = 10) -> dict:
    """
    Tool function that processes parameters.
    
    Args:
        param1: Required string parameter
        param2: Optional integer with default value
        
    Returns:
        Dictionary with results
    """
    # Your implementation here
    result = f"Processed {param1} with value {param2}"
    
    return {
        "status": "completed",
        "result": result,
        "param1": param1,
        "param2": param2
    }
```

### Async Tool Example

```python
@tool(name="async_tool", description="Async operation example")
async def async_operation(url: str) -> dict:
    """
    Example of an async tool that makes HTTP requests.
    """
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return {
                "url": url,
                "status_code": response.status,
                "content_length": len(await response.text())
            }
```

### Tool Best Practices

1. **Clear Naming**: Use descriptive names for tools and parameters
2. **Type Hints**: Always include type hints for better documentation
3. **Error Handling**: Handle exceptions gracefully within your tools
4. **Return Dictionaries**: Always return JSON-serializable dictionaries
5. **Documentation**: Include docstrings explaining what the tool does
6. **Security**: Validate inputs and avoid executing arbitrary code

## Built-in Tools

The system comes with several example tools:

### echo
Echoes back the input message.

**Request:**
```json
{
  "name": "echo",
  "message": "Hello, World!"
}
```

**Response:**
```json
{
  "original_message": "Hello, World!",
  "echo": "Hello, World!",
  "timestamp": "1234567890.123"
}
```

### system_info
Returns basic system information.

**Request:**
```json
{
  "name": "system_info"
}
```

**Response:**
```json
{
  "platform": "Linux",
  "platform_version": "...",
  "hostname": "my-computer",
  "python_version": "3.10.0",
  "cpu_count": 8,
  "current_directory": "/home/user"
}
```

### calculate
Safely evaluates mathematical expressions.

**Request:**
```json
{
  "name": "calculate",
  "expression": "2 + 2 * 3"
}
```

**Response:**
```json
{
  "expression": "2 + 2 * 3",
  "result": 8,
  "success": true
}
```

### file_operations
Performs basic file operations.

**Request:**
```json
{
  "name": "file_operations",
  "operation": "read",
  "path": "example.txt"
}
```

**Operations:**
- `read` - Read file contents
- `write` - Write content to file (requires `content` parameter)
- `exists` - Check if file/directory exists
- `list` - List directory contents

### template
A flexible template function that accepts any arguments.

**Request:**
```json
{
  "name": "template",
  "custom_arg1": "value1",
  "custom_arg2": 123,
  "nested": {"key": "value"}
}
```

## Security Considerations

1. **Input Validation**: Always validate and sanitize inputs in your tool functions
2. **Access Control**: Consider implementing access control for sensitive operations
3. **File System Access**: Be cautious with file operations and path traversal
4. **Command Execution**: Avoid executing shell commands or arbitrary code
5. **Rate Limiting**: Consider implementing rate limiting for resource-intensive operations
6. **Logging**: Log all tool executions for audit purposes

## Troubleshooting

### Tools Not Found

If your tools aren't being recognized:
1. Ensure they're decorated with `@tool(name="...")`
2. Check that `local_tool_responses.py` is in the same directory
3. Look for import errors in the logs

### Data Channel Not Connecting

1. Verify the data channel name matches on both sides
2. Check LiveKit room permissions
3. Ensure the participant has data channel privileges

### Tool Execution Errors

1. Check the logs for detailed error messages
2. Verify the tool function signature matches the arguments being sent
3. Test tools locally using `--test-tools` mode

## Advanced Usage

### Custom Argument Formatting

The `_format_tool_arguments` method in `LightberryToolServer` can be customized to handle special argument transformations:

```python
def _format_tool_arguments(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Custom argument formatting logic."""
    args = super()._format_tool_arguments(tool_call)
    
    # Add custom transformations here
    # For example, convert string "true"/"false" to boolean
    for key, value in args.items():
        if isinstance(value, str) and value.lower() in ["true", "false"]:
            args[key] = value.lower() == "true"
    
    return args
```

### Extending the System

To extend the system:

1. **Custom Response Formats**: Modify the response structure in `process_tool_call`
2. **Authentication**: Add authentication checks before executing tools
3. **Tool Categories**: Organize tools into categories with different permissions
4. **Result Caching**: Implement caching for expensive operations
5. **Tool Chaining**: Allow tools to call other tools

## Example Integration

Here's an example of how to send tool calls from another LiveKit participant:

```javascript
// JavaScript/TypeScript example
const dataChannel = await room.localParticipant.publishData(
  JSON.stringify({
    name: "echo",
    message: "Hello from the other side!"
  }),
  { 
    reliable: true,
    destination: "tool-calls"  // Must match the data channel name
  }
);

// Listen for responses
room.on('dataReceived', (data, participant) => {
  const response = JSON.parse(new TextDecoder().decode(data));
  console.log('Tool response:', response);
});
```

## Performance Considerations

1. **Async Operations**: Use async functions for I/O operations
2. **Resource Management**: Clean up resources properly in tool functions
3. **Timeout Handling**: Consider implementing timeouts for long-running operations
4. **Queue Management**: The system uses a queue to handle concurrent tool calls
5. **Error Recovery**: Tools should be idempotent when possible

## Conclusion

The Lightberry Tool System provides a flexible and secure way to execute local functions via LiveKit's data channel. By following the patterns and best practices outlined in this documentation, you can create powerful integrations between remote agents and local devices.