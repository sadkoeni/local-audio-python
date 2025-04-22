# Python Local Audio Device Example

A basic example of using local microphone and audio output devices using the LiveKit [Python SDK](https://github.com/livekit/python-sdks).

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
cd local-audio-python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up the environment by copying `.env.example` to `.env` and filling in the required values:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`

You can also do this automatically using the LiveKit CLI:

```bash
lk app env
```

Run the agent:

```console
python3 agent.py dev
```

This agent requires a frontend application to communicate with. You can use one of our example frontends in [livekit-examples](https://github.com/livekit-examples/), create your own following one of our [client quickstarts](https://docs.livekit.io/realtime/quickstarts/), or test instantly against one of our hosted [Sandbox](https://cloud.livekit.io/projects/p_/sandbox) frontends.
