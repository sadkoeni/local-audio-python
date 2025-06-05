#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "livekit",
#   "livekit_api",
#   "pyaudio",
#   "pyserial",
#   "python-dotenv",
#   "asyncio",
# ]
# ///

import os
import logging
import asyncio
from dotenv import load_dotenv
from signal import SIGINT, SIGTERM
from livekit import rtc
import pyaudio
import numpy as np
from auth import generate_token
from collections import deque

load_dotenv()
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set in your .env file
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

# default audio settings
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
CHUNK_SIZE = 480  # 10ms at 48kHz

def find_audio_devices():
    """Find suitable input and output audio devices"""
    audio = pyaudio.PyAudio()
    
    input_device = None
    output_device = None
    
    logger = logging.getLogger(__name__)
    logger.info("Available audio devices:")
    
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        logger.info(f"Device {i}: {info['name']} - Max Inputs: {info['maxInputChannels']}, Max Outputs: {info['maxOutputChannels']}")
        
        # Find a suitable input device (microphone)
        if input_device is None and info['maxInputChannels'] >= NUM_CHANNELS:
            try:
                # Test if we can open the device
                test_stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=NUM_CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=CHUNK_SIZE
                )
                test_stream.close()
                input_device = i
                logger.info(f"Selected input device {i}: {info['name']}")
            except Exception as e:
                logger.debug(f"Could not use device {i} for input: {e}")
        
        # Find a suitable output device (speakers)
        if output_device is None and info['maxOutputChannels'] >= NUM_CHANNELS:
            try:
                # Test if we can open the device
                test_stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=NUM_CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    output_device_index=i,
                    frames_per_buffer=CHUNK_SIZE
                )
                test_stream.close()
                output_device = i
                logger.info(f"Selected output device {i}: {info['name']}")
            except Exception as e:
                logger.debug(f"Could not use device {i} for output: {e}")
    
    audio.terminate()
    return input_device, output_device

class SimpleAEC:
    """Simple Acoustic Echo Cancellation using adaptive filtering"""
    
    def __init__(self, filter_length=1024, step_size=0.01):
        self.filter_length = filter_length
        self.step_size = step_size
        self.w = np.zeros(filter_length)  # Adaptive filter coefficients
        self.x_buffer = deque(maxlen=filter_length)  # Reference signal buffer (speaker output)
        self.y_buffer = deque(maxlen=filter_length)  # Error signal buffer
        
        # Initialize buffers with zeros
        for _ in range(filter_length):
            self.x_buffer.append(0.0)
            self.y_buffer.append(0.0)
    
    def process(self, microphone_signal, reference_signal):
        """
        Process audio frame with AEC
        microphone_signal: audio from microphone (numpy array)
        reference_signal: audio being played to speakers (numpy array)
        Returns: echo-cancelled microphone signal
        """
        if len(microphone_signal) == 0:
            return microphone_signal
        
        output = np.zeros_like(microphone_signal, dtype=np.float32)
        
        for i in range(len(microphone_signal)):
            # Add reference signal to buffer
            if len(reference_signal) > i:
                self.x_buffer.append(float(reference_signal[i]))
            else:
                self.x_buffer.append(0.0)
            
            # Convert buffer to numpy array for processing
            x_vec = np.array(list(self.x_buffer), dtype=np.float32)
            
            # Normalize the reference vector to prevent instability
            x_norm = np.linalg.norm(x_vec)
            if x_norm > 1e-10:  # Avoid division by zero
                x_vec_normalized = x_vec / x_norm
            else:
                x_vec_normalized = x_vec
            
            # Estimate echo using adaptive filter
            estimated_echo = np.dot(self.w, x_vec_normalized)
            
            # Clamp estimated echo to reasonable range
            estimated_echo = np.clip(estimated_echo, -32000, 32000)
            
            # Subtract estimated echo from microphone signal
            error_signal = float(microphone_signal[i]) - estimated_echo
            
            # Clamp error signal to valid audio range
            error_signal = np.clip(error_signal, -32000, 32000)
            output[i] = error_signal
            
            # Update filter coefficients using normalized LMS algorithm
            if x_norm > 1e-10:
                # Normalized LMS update with regularization
                mu_normalized = self.step_size / (x_norm + 1e-6)
                self.w += mu_normalized * error_signal * x_vec_normalized
                
                # Prevent filter coefficients from growing too large
                self.w = np.clip(self.w, -10.0, 10.0)
            
            # Check for invalid values and reset if necessary
            if np.any(np.isnan(self.w)) or np.any(np.isinf(self.w)):
                self.w = np.zeros_like(self.w)
            
            # Add error to buffer for next iteration
            self.y_buffer.append(error_signal)
        
        # Final clipping and conversion to int16
        output = np.clip(output, -32767, 32767)
        
        # Check for any remaining invalid values
        output = np.where(np.isfinite(output), output, 0.0)
        
        return output.astype(np.int16)

async def main(room: rtc.Room):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Find suitable audio devices
    input_device_index, output_device_index = find_audio_devices()
    
    if input_device_index is None:
        logger.error("No suitable input device found!")
        return
    
    if output_device_index is None:
        logger.error("No suitable output device found!")
        return

    # Create the audio source for publishing microphone audio
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    
    # Initialize AEC
    aec = SimpleAEC(filter_length=512, step_size=0.001)
    
    # Flag to control when to stop capturing
    running = True
    
    # Buffer for reference signal (what we're playing to speakers)
    current_reference = np.zeros(CHUNK_SIZE, dtype=np.int16)

    # init pyaudio
    audio = pyaudio.PyAudio()
    
    # Audio input stream for microphone
    input_stream = audio.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK_SIZE,
    )
    
    # Audio output stream for speakers
    output_stream = audio.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        output_device_index=output_device_index
    )

    # Task to handle audio capture from microphone
    async def capture_audio():
        try:
            while running:
                # Read audio data from microphone
                try:
                    in_data = input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    logger.warning(f"Error reading audio: {e}")
                    await asyncio.sleep(0.01)
                    continue
                
                # Process the audio data
                input_array = np.frombuffer(in_data, dtype=np.int16)
                
                # Apply AEC - use current reference signal being played
                processed_audio = aec.process(input_array, current_reference)
                
                # Create an audio frame
                audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, CHUNK_SIZE)
                
                # Convert the audio frame to a numpy array
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
                
                # Copy processed (echo-cancelled) audio
                sample_count = min(len(processed_audio), len(audio_data))
                np.copyto(audio_data[:sample_count], processed_audio[:sample_count])
                
                # Publish the frame
                await source.capture_frame(audio_frame)
                
                # Small sleep to not overwhelm CPU
                await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in audio capture: {e}")
        finally:
            logger.info("Audio capture stopped")

    # Function to handle received audio frames from other participants
    async def receive_audio_frames(stream: rtc.AudioStream):
        nonlocal current_reference
        async for frame in stream:
            try:
                # Get the audio data
                audio_data = frame.frame.data.tobytes()
                
                # Convert to numpy array for AEC reference
                reference_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Update reference signal for AEC
                if len(reference_array) == CHUNK_SIZE:
                    current_reference = reference_array.copy()
                else:
                    # Pad or truncate to match chunk size
                    if len(reference_array) < CHUNK_SIZE:
                        current_reference = np.pad(reference_array, (0, CHUNK_SIZE - len(reference_array)))
                    else:
                        current_reference = reference_array[:CHUNK_SIZE]
                
                # Play the audio
                output_stream.write(audio_data)
                
            except Exception as e:
                logger.warning(f"Error writing audio output: {e}")

    # Event handler for when a new participant connects
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("participant connected: %s %s", participant.sid, participant.identity)

    # Event handler for when we're subscribed to a new track
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            asyncio.ensure_future(receive_audio_frames(audio_stream))

    # Event handler for when a track is published by another participant
    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    # Generate LiveKit token and connect to room
    token = generate_token(ROOM_NAME, "audio-streamer2", "Audio Streamer")
    await room.connect(LIVEKIT_URL, token)
    logger.info("connected to room %s", room.name)
    
    # Publish microphone track
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logger.info("published track %s", publication.sid)
    
    # Start the audio capture task
    audio_capture_task = asyncio.create_task(capture_audio())
    
    # Keep the main task running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Main task cancelled, cleaning up...")
        running = False
        audio_capture_task.cancel()
        
        # Clean up audio streams
        try:
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            audio.terminate()
        except Exception as e:
            logger.warning(f"Error during audio cleanup: {e}")
        
        await room.disconnect()


async def async_main():
    """Main async function to replace the old event loop approach"""
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("stream_audio.log"),
            logging.StreamHandler(),
        ],
    )
    
    room = rtc.Room()
    
    # Setup signal handlers for cleanup
    async def cleanup():
        try:
            await room.disconnect()
        except:
            pass

    # Handle shutdown gracefully
    try:
        await main(room)
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    finally:
        await cleanup()


if __name__ == "__main__":
    # Use the modern asyncio.run() instead of deprecated get_event_loop()
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
    except Exception as e:
        logging.error(f"Application error: {e}")
        raise 