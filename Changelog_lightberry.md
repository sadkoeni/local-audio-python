# Changelog - Lightberry Project

This changelog tracks all significant changes made to the Lightberry audio streaming project.

## Format

Each entry includes:
- **id**: Unique identifier for the change
- **timestamp**: ISO-8601 GMT timestamp when the change was made
- **description**: What was changed and why
- **method**: How the change was implemented
- **outcome**: Result of the change (filled after user confirmation)

---

### Fixed LiveKit DataChannel Error

* id: fix-livekit-datachannel-001
* timestamp: 2025-01-23T22:30:00Z
* description: Fixed "module 'livekit.rtc' has no attribute 'DataChannel'" error by updating the code to use the correct LiveKit Python SDK API for data communication
* method: Replaced references to rtc.DataChannel and rtc.DataChannelMessage with the correct LiveKit data API using room.local_participant.publish_data() for sending and room.on("data_received") for receiving data packets. The SDK uses DataPacket objects with topics for data communication rather than dedicated DataChannel objects.
* outcome: null