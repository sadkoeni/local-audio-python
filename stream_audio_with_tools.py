#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "livekit",
#   "livekit-api",
#   "sounddevice",
#   "python-dotenv",
#   "asyncio",
#   "numpy",
# ]
# ///
"""
Modified audio streaming client with tool call support via LiveKit data channel.

This version adds:
- Data channel support for receiving tool calls
- Integration with LightberryToolServer
- Sending tool execution results back via data channel
"""

import os
import logging
import asyncio
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from livekit import rtc
from lightberry_tool_server import LightberryToolServer

# Import the original stream_audio module
import stream_audio

load_dotenv()

logger = logging.getLogger(__name__)


class AudioStreamWithTools(stream_audio.AudioStreamer):
    """Extended AudioStreamer with data channel support for tool calls."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_server: Optional[LightberryToolServer] = None
        self.room: Optional[rtc.Room] = None
        self.data_channel_name: Optional[str] = None
        
    def set_tool_server(self, tool_server: LightberryToolServer):
        """Set the tool server instance."""
        self.tool_server = tool_server
        logger.info("Tool server connected to audio streamer")
        
    def set_data_channel_name(self, name: str):
        """Set the data channel name to use."""
        self.data_channel_name = name
        if self.tool_server:
            self.tool_server.set_data_channel_name(name)
        logger.info(f"Data channel name set to: {name}")
        
    def set_room(self, room: rtc.Room):
        """Set the LiveKit room instance."""
        self.room = room
        logger.info("Room connected to audio streamer")


async def main_with_tools(
    participant_name: str = "python-user",
    device_index: Optional[int] = None,
    enable_aec: bool = True,
    data_channel_name: Optional[str] = None
):
    """
    Main function with tool support via data channel.
    
    Args:
        participant_name: Name of the participant
        device_index: Audio device index to use
        enable_aec: Whether to enable echo cancellation
        data_channel_name: Name of the data channel for tool calls
    """
    # Create extended audio streamer
    streamer = AudioStreamWithTools(device_index, enable_aec)
    
    # Create and configure tool server
    tool_server = LightberryToolServer(data_channel_name)
    streamer.set_tool_server(tool_server)
    
    if data_channel_name:
        streamer.set_data_channel_name(data_channel_name)
    
    # Store the event loop reference
    streamer.loop = asyncio.get_running_loop()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lightberry_audio_tools.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting audio streaming with tools support")
    logger.info(f"Participant name: {participant_name}")
    logger.info(f"Device index: {device_index}")
    logger.info(f"Echo cancellation: {enable_aec}")
    logger.info(f"Data channel: {data_channel_name}")
    
    # Create LiveKit room
    room = rtc.Room()
    streamer.set_room(room)
    
    # Setup data received handler
    async def handle_data_received(data_packet: rtc.DataPacket):
        """Handle data received from LiveKit data channel."""
        try:
            # Check if this is for our data channel (by topic)
            if data_channel_name and data_packet.topic == data_channel_name:
                # Decode the data
                message = data_packet.data.decode('utf-8')
                logger.debug(f"Received data channel message: {message}")
                
                # Pass to tool server for processing
                if streamer.tool_server:
                    await streamer.tool_server.handle_data_message(message)
                else:
                    logger.warning("Tool server not configured, ignoring data channel message")
            
        except Exception as e:
            logger.error(f"Error handling data channel message: {e}")
    
    # Setup response callback to send results back
    async def send_tool_response(response: Dict[str, Any]):
        """Send tool execution results back via data channel."""
        try:
            if streamer.room and streamer.room.local_participant:
                response_json = json.dumps(response)
                await streamer.room.local_participant.publish_data(
                    response_json.encode('utf-8'),
                    kind=rtc.DataPacketKind.KIND_RELIABLE,
                    topic=data_channel_name
                )
                logger.debug(f"Sent tool response: {response_json}")
            else:
                logger.warning("Room or local participant not available")
        except Exception as e:
            logger.error(f"Error sending tool response: {e}")
    
    # Configure tool server response callback
    tool_server.set_response_callback(send_tool_response)
    
    @room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        """Handle data received event."""
        asyncio.create_task(handle_data_received(data_packet))
    
    # Copy event handlers from original stream_audio
    @room.on("track_published")
    def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info("track published: %s from %s", publication.sid, participant.identity)

    @room.on("track_unpublished")
    def on_track_unpublished(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info("track unpublished: %s from %s", publication.sid, participant.identity)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info("track subscribed: %s from %s", publication.sid, participant.identity)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(process_audio_stream(audio_stream, participant))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info("track unsubscribed: %s from %s", publication.sid, participant.identity)

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("participant connected: %s %s", participant.sid, participant.identity)
        with streamer.participants_lock:
            streamer.participants[participant.sid] = {
                'name': participant.identity or f"User_{participant.sid[:8]}",
                'db_level': stream_audio.INPUT_DB_MIN,
                'last_update': stream_audio.time.time()
            }

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info("participant disconnected: %s %s", participant.sid, participant.identity)
        with streamer.participants_lock:
            if participant.sid in streamer.participants:
                del streamer.participants[participant.sid]

    @room.on("connected")
    def on_connected():
        logger.info("Successfully connected to LiveKit room")

    @room.on("disconnected")
    def on_disconnected(reason):
        logger.info(f"Disconnected from LiveKit room: {reason}")
    
    # Audio processing task
    async def audio_processing_task():
        """Process incoming audio frames."""
        logger.info("Audio processing task started")
        while streamer.running:
            try:
                audio_frame = await asyncio.wait_for(
                    streamer.audio_input_queue.get(), 
                    timeout=1.0
                )
                await streamer.source.capture_frame(audio_frame)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                
    # Meter display task
    async def meter_task():
        """Update the audio level meter display."""
        logger.info("Meter display task started")
        last_update = stream_audio.time.time()
        
        while streamer.running:
            try:
                current_time = stream_audio.time.time()
                if current_time - last_update >= 1.0 / stream_audio.FPS:
                    streamer.draw_ui()
                    last_update = current_time
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error updating meter: {e}")
                
    # Audio stream processing
    async def process_audio_stream(audio_stream: rtc.AudioStream, participant: rtc.RemoteParticipant):
        """Process audio stream from a remote participant."""
        logger.info(f"Starting audio processing for participant: {participant.identity}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                if streamer.audio_processor:
                    streamer.audio_processor.process_reverse_stream(event.frame)
                    
                # Update participant volume
                frame_data = stream_audio.np.frombuffer(event.frame.data, dtype=stream_audio.np.int16)
                rms = stream_audio.np.sqrt(stream_audio.np.mean(frame_data.astype(stream_audio.np.float32) ** 2))
                max_int16 = stream_audio.np.iinfo(stream_audio.np.int16).max
                db_level = 20.0 * stream_audio.np.log10(rms / max_int16 + 1e-6)
                
                with streamer.participants_lock:
                    if participant.sid in streamer.participants:
                        streamer.participants[participant.sid]['db_level'] = db_level
                        streamer.participants[participant.sid]['last_update'] = stream_audio.time.time()
    
    try:
        # Start tool server
        await tool_server.start()
        
        # Start audio devices
        logger.info("Starting audio devices...")
        streamer.start_audio_devices()
        
        # Start keyboard handler
        logger.info("Starting keyboard handler...")
        streamer.start_keyboard_handler()
        
        # Initialize terminal for stable UI
        streamer.init_terminal()
        
        # Connect to LiveKit room
        logger.info("Connecting to LiveKit room...")
        token = stream_audio.generate_token(
            stream_audio.ROOM_NAME, 
            participant_name, 
            participant_name
        )
        
        await room.connect(stream_audio.LIVEKIT_URL, token)
        logger.info("connected to room %s", room.name)
        
        # Publish microphone track
        logger.info("Publishing microphone track...")
        track = rtc.LocalAudioTrack.create_audio_track("mic", streamer.source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await room.local_participant.publish_track(track, options)
        logger.info("published track %s", publication.sid)
        
        # Start background tasks
        logger.info("Starting background tasks...")
        audio_task = asyncio.create_task(audio_processing_task())
        meter_display_task = asyncio.create_task(meter_task())
        
        logger.info("=== Audio streaming with tools started. Press Ctrl+C to stop. ===")
        
        # Keep running until interrupted
        try:
            while streamer.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping audio streaming...")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        logger.info("Starting cleanup...")
        streamer.running = False
        
        # Stop tool server
        await tool_server.stop()
        
        if 'audio_task' in locals():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
        
        if 'meter_display_task' in locals():
            meter_display_task.cancel()
            try:
                await meter_display_task
            except asyncio.CancelledError:
                pass
        
        streamer.stop_audio_devices()
        await room.disconnect()
        await asyncio.sleep(0.5)
        
        streamer.cleanup_terminal()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LiveKit Audio Streaming with Tool Support")
    parser.add_argument(
        "--name", 
        type=str, 
        default="python-user",
        help="Participant name"
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
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_tools:
        # List available tools
        from local_tool_responses import get_available_tools
        tools = get_available_tools()
        print("Available tools:")
        for name, info in tools.items():
            print(f"  - {name}: {info['description']}")
        exit(0)
    
    # Run with tools support
    asyncio.run(main_with_tools(
        participant_name=args.name,
        device_index=args.device,
        enable_aec=not args.no_aec,
        data_channel_name=args.data_channel
    ))