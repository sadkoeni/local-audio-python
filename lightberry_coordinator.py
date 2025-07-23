#!/usr/bin/env python3
"""
Lightberry Coordinator

Main script that coordinates the audio streaming client with tool execution support.
This script starts both the audio client and the tool server, managing their lifecycle.
"""

import asyncio
import argparse
import logging
import sys
from typing import Optional

# Import our enhanced audio streaming module
from stream_audio_with_tools import main_with_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_lightberry(
    participant_name: str = "lightberry-user",
    device_index: Optional[int] = None,
    enable_aec: bool = True,
    data_channel_name: str = "tool-calls",
    log_level: str = "INFO"
):
    """
    Run the Lightberry system with audio streaming and tool execution.
    
    Args:
        participant_name: Name to use in the LiveKit room
        device_index: Audio device index (None for default)
        enable_aec: Whether to enable echo cancellation
        data_channel_name: Name of the data channel for tool calls
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    logger.info("Starting Lightberry Coordinator")
    logger.info(f"Configuration:")
    logger.info(f"  Participant: {participant_name}")
    logger.info(f"  Audio Device: {device_index if device_index is not None else 'Default'}")
    logger.info(f"  Echo Cancellation: {'Enabled' if enable_aec else 'Disabled'}")
    logger.info(f"  Data Channel: {data_channel_name}")
    logger.info(f"  Log Level: {log_level}")
    
    try:
        # Run the main audio streaming with tools
        await main_with_tools(
            participant_name=participant_name,
            device_index=device_index,
            enable_aec=enable_aec,
            data_channel_name=data_channel_name
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error running Lightberry: {e}")
        raise
    finally:
        logger.info("Lightberry Coordinator stopped")


def main():
    """Main entry point for the coordinator."""
    parser = argparse.ArgumentParser(
        description="Lightberry - LiveKit Audio Streaming with Tool Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python lightberry_coordinator.py
  
  # Use a specific audio device
  python lightberry_coordinator.py --device 2
  
  # Disable echo cancellation
  python lightberry_coordinator.py --no-aec
  
  # Use custom data channel name
  python lightberry_coordinator.py --data-channel my-tools
  
  # Enable debug logging
  python lightberry_coordinator.py --log-level DEBUG
  
  # List available audio devices
  python list_devices.py
  
  # List available tools
  python lightberry_coordinator.py --list-tools
        """
    )
    
    parser.add_argument(
        "--name", 
        type=str, 
        default="lightberry-user",
        help="Participant name in the LiveKit room"
    )
    
    parser.add_argument(
        "--device", 
        type=int, 
        default=None,
        help="Audio device index (run list_devices.py to see available devices)"
    )
    
    parser.add_argument(
        "--no-aec", 
        action="store_true",
        help="Disable echo cancellation"
    )
    
    parser.add_argument(
        "--data-channel",
        type=str,
        default="tool-calls",
        help="Name of the data channel for tool calls (default: tool-calls)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )
    
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Run tool server test mode"
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.list_tools:
        # List available tools
        from local_tool_responses import get_available_tools
        tools = get_available_tools()
        
        print("\n" + "="*60)
        print("AVAILABLE TOOLS")
        print("="*60)
        
        for name, info in tools.items():
            print(f"\n📦 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Function: {info['function']}")
            print(f"   Module: {info['module']}")
        
        print("\n" + "="*60)
        print(f"Total tools available: {len(tools)}")
        print("="*60 + "\n")
        
        return
    
    if args.test_tools:
        # Run tool server in test mode
        from lightberry_tool_server import LightberryToolServer
        import json
        
        async def test_tools():
            server = LightberryToolServer()
            
            print("\n" + "="*60)
            print("TOOL SERVER TEST MODE")
            print("="*60)
            
            # Test various tool calls
            test_calls = [
                {"name": "echo", "message": "Testing Lightberry!"},
                {"name": "system_info"},
                {"name": "calculate", "expression": "2 + 2"},
                {"name": "template", "test_arg": "value", "number": 42}
            ]
            
            for call in test_calls:
                print(f"\n🔧 Testing tool: {call['name']}")
                print(f"   Input: {json.dumps(call, indent=2)}")
                
                response = await server.process_tool_call(json.dumps(call))
                
                if response['status'] == 'success':
                    print(f"   ✅ Success!")
                    print(f"   Output: {json.dumps(response['result'], indent=2)}")
                else:
                    print(f"   ❌ Error: {response['error']}")
                    print(f"   Details: {response['details']}")
            
            print("\n" + "="*60)
        
        asyncio.run(test_tools())
        return
    
    # Run the main coordinator
    try:
        asyncio.run(run_lightberry(
            participant_name=args.name,
            device_index=args.device,
            enable_aec=not args.no_aec,
            data_channel_name=args.data_channel,
            log_level=args.log_level
        ))
    except KeyboardInterrupt:
        print("\n\nShutdown complete. Goodbye! 👋")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()