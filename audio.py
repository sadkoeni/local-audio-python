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
import argparse
from dotenv import load_dotenv
from signal import SIGINT, SIGTERM
from livekit import rtc
import pyaudio
import numpy as np
from auth import generate_token

load_dotenv()
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set in your .env file
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

# default audio settings
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
CHUNK_SIZE = 480  # 10ms at 48kHz
OUTPUT_CHANNELS = 2  # Speakers support 2 channels

async def main(room: rtc.Room, participant_name: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create the audio source for publishing
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    
    # Flag to control when to stop capturing
    running = True

    # init pyaudio
    audio = pyaudio.PyAudio()
    
    # Input stream for microphone (device 0: DC Microphone)
    input_stream = audio.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=1,
    )
    
    # Output stream for speakers (device 2: MacBook Pro Speakers)
    output_stream = audio.open(
        format=pyaudio.paInt16,
        channels=OUTPUT_CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        output_device_index=2
    )

    # Task to handle audio capture and publishing
    async def capture_and_publish_audio():
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
                
                # Create an audio frame
                audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, CHUNK_SIZE)
                
                # Convert the audio frame to a numpy array
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
                
                # Copy only as many samples as we have in the input or can fit in the output
                sample_count = min(len(input_array), len(audio_data))
                np.copyto(audio_data[:sample_count], input_array[:sample_count])
                
                # Capture the frame for publishing
                await source.capture_frame(audio_frame)
                
                # Small sleep to not overwhelm CPU
                await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in audio capture: {e}")
        finally:
            logger.info("Audio capture stopped")

    # Function to handle received audio frames
    async def receive_audio_frames(stream: rtc.AudioStream):
        async for frame in stream:
            # Convert mono audio to stereo for speakers
            mono_data = frame.frame.data.tobytes()
            mono_array = np.frombuffer(mono_data, dtype=np.int16)
            
            # Duplicate mono channel to create stereo
            stereo_array = np.repeat(mono_array, 2)
            
            output_stream.write(stereo_array.tobytes())

    # Event handlers
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

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("participant connected: %s %s", participant.sid, participant.identity)

    # Generate LiveKit token and connect
    token = generate_token(ROOM_NAME, participant_name, participant_name)
    await room.connect(LIVEKIT_URL, token)
    logger.info("connected to room %s", room.name)
    
    # Publish microphone track
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logger.info("published track %s", publication.sid)
    
    # Start the audio capture task
    audio_capture_task = asyncio.create_task(capture_and_publish_audio())
    
    # Keep the main task running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        running = False
        audio_capture_task.cancel()
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        audio.terminate()
        await room.disconnect()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LiveKit bidirectional audio client")
    parser.add_argument(
        "--name", 
        "-n",
        type=str,
        default="audio-participant",
        help="Participant name to use when connecting to the room (default: audio-participant)"
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("audio.log"),
            logging.StreamHandler(),
        ],
    )
    
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        try:
            await room.disconnect()
        except:
            pass
        task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not task]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    asyncio.ensure_future(main(room, args.name))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close() 