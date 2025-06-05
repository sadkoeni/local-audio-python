#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "pyaudio",
# ]
# ///

import pyaudio

p = pyaudio.PyAudio()

print("Available Audio Devices:")
print("-" * 60)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
    print(f"  Max Input Channels: {info['maxInputChannels']}")
    print(f"  Max Output Channels: {info['maxOutputChannels']}")
    print(f"  Default Sample Rate: {info['defaultSampleRate']}")
    print("-" * 60)

p.terminate()