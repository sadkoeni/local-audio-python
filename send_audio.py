import os
import logging
import asyncio
from dotenv import load_dotenv
from signal import SIGINT, SIGTERM
from livekit import rtc, api
import pyaudio
import numpy as np
from queue import Queue
from auth import generate_token

load_dotenv()
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set in your .env file
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

# default audio settings
SAMPLE_RATE = 48000
NUM_CHANNELS = 1

async def main(room: rtc.Room):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create the audio source before using it in the callback
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    
    # Create a queue to pass audio frames from callback to async context
    frame_queue = Queue()
    running = True

    # init audio stream
    audio = pyaudio.PyAudio()

    def stream_callback(in_data, frame_count, time_info, status):
        samples_per_channel = 480  # 10ms at 48kHz
        # First convert the incoming data to int16 numpy array
        input_array = np.frombuffer(in_data, dtype=np.int16)
        # Create an audio frame with the same sample rate and number of channels as the input
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
        # Convert the audio frame to a numpy array
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
        # Copy only as many samples as we have in the input or can fit in the output
        sample_count = min(len(input_array), len(audio_data))
        np.copyto(audio_data[:sample_count], input_array[:sample_count])
        
        # add to queue instead
        if running:
            frame_queue.put(audio_frame)
        return (None, pyaudio.paContinue)
    
    # Start recording audio
    audio_stream = audio.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        stream_callback=stream_callback)

    # Task to process audio frames from the queue
    async def process_audio_frames():
        while running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                await source.capture_frame(frame)
            await asyncio.sleep(0.001)  # Short sleep to avoid CPU spinning

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("participant connected: %s %s", participant.sid, participant.identity)

    # By default, autosubscribe is enabled. The participant will be subscribed to
    # all published tracks in the room

    # generate LiveKit token
    token = generate_token(ROOM_NAME, "audio-publisher", "Audio Publisher")

    await room.connect(LIVEKIT_URL, token)
    logger.info("connected to room %s", room.name)
    
    # publish a track
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logging.info("published track %s", publication.sid)
    
    # Start processing audio frames
    frame_processor = asyncio.create_task(process_audio_frames())
    
    # Keep the main task running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        running = False
        frame_processor.cancel()
        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()
        await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("send_audio.log"),
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

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
