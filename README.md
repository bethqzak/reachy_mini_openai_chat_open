---
title: Reachy Mini OpenAI Chat (Open)
emoji: 🤖
colorFrom: purple
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Voice chat with your Reachy Mini using your own OpenAI key
tags:
  - reachy_mini_python_app
---

# Reachy Mini · OpenAI Chat (Open) 🤖

Talk with your **Reachy Mini** using **your own OpenAI key**. This is a lean,
standalone Reachy Mini app that reproduces the feel of the official
conversation app, but swaps the brain for the **OpenAI Realtime API**
(`gpt-realtime`) — voice in, voice out, over a single low-latency WebSocket.

This is the **open edition**: no chat logging, no telemetry, no fleet
management. Conversations are never written to disk and never leave the robot
(audio and camera frames go only to OpenAI, under your own key).

Everything that makes the robot feel alive is here:

- 🎙️ **Live voice conversation** — real-time speech-to-speech, natural turn-taking and barge-in (server-side VAD).
- 💫 **Ambient movement** — gentle "breathing" bob and idle drift so it's never frozen.
- 🗣️ **Speech-reactive motion** — head and antennas wobble in time with its own voice.
- 😄 **Expressive gestures & emotions** — the model moves to express itself (nod, shake, tilt, wiggle antennas, happy, excited, sad, surprised, sleepy, dance) via a function-calling tool.
- 👀 **Camera vision / image recognition** — ask "what do you see?", "what is this?", or "read this to me" and it captures a frame and describes what's actually in front of it.
- 🙂 **Optional face tracking** — a local OpenCV detector lets it turn its head to follow the nearest face; the model can switch it on and off.
- 🎛️ **Settings panel in the dashboard** — a web page (served at `http://localhost:8042`) to change voice/personality, toggle behaviours live, test gestures, and watch the live transcript (kept in memory only).

You pay OpenAI directly for usage. Apache-2.0 licensed.

---

## Design decisions

- **OpenAI Realtime API, not a pipeline.** The brain is the `gpt-realtime` speech-to-speech model over a single WebSocket — chosen over a Whisper → GPT → TTS pipeline for the lowest latency, native tool-calling to drive movement, and built-in image input for vision. It's the closest match to the official app's live feel.
- **Lean standalone app, not a fork.** Rather than forking the official conversation app, this is a clean, dependency-light build. Gestures and emotions are procedural (no external motion libraries to download), which keeps it easy to read and modify.
- **Nothing is recorded.** The live transcript in the settings page is a small in-memory buffer that vanishes when the app stops. No files, no uploads, no analytics. Faces are pixelated locally before a camera frame is sent to OpenAI (`REACHY_BLUR_FACES=true` by default).
- **Dashboard settings panel.** A small web UI is served so voice, personality, behaviours, gesture tests and the live transcript are all adjustable without editing files. See [Settings page](#settings-page-in-the-dashboard).
- **Targets a physical Reachy Mini Lite.** Runs on the robot via the daemon on your host computer, and works the same whether started from the desktop dashboard or the Reachy Mini phone app. See [Running from your computer or your phone](#running-from-your-computer-or-your-phone).

---

## 1. Requirements

- A Reachy Mini (Lite or Wireless) with the daemon running (`reachy-mini` 1.8.4 or newer; tested with 1.11.0).
- An **OpenAI API key** with **Realtime API** access.
- Python 3.11+ (`reachy-mini` 1.9 and later require it; 3.10 works only with `reachy-mini` 1.8.4).

Daemon 1.8.4 through 1.11 are supported (last verified against `reachy-mini`
1.11.0, released 21 Sept 2026). Intel Macs top out
at 1.8.4 because `reachy-mini` 1.9 and later hard-require `onnxruntime==1.27.0`,
which has no x86_64 macOS wheels; this app sets no upper bound, so pip resolves
1.8.4 there and the SDK calls it makes are identical. Daemon 1.11 dropped
`scipy` from the SDK's own dependencies; this app declares its own `scipy` (for
audio resampling), so nothing changes for you.

## 2. Install on the robot

**From the dashboard (easiest):** open the Reachy Mini dashboard, go to the app
store / install-from-URL, and install this Space's URL. Then it appears in your
app list.

**Manually (wired / Lite):**

```bash
uv pip install -e /path/to/reachy_mini_openai_chat_open
# or:  pip install -e /path/to/reachy_mini_openai_chat_open
```

**Manually (Wireless, over the network):**

```bash
scp -r reachy_mini_openai_chat_open pollen@reachy-mini.local:/tmp/reachy_mini_openai_chat_open
ssh pollen@reachy-mini.local \
  "/venvs/apps_venv/bin/pip install /tmp/reachy_mini_openai_chat_open"
```

## 3. Add your OpenAI key

**Easiest: use the app.** Start the app (step 4), open the settings page at
`http://localhost:8042`, and paste your key into the **OpenAI API key** box.
The key is stored on the robot (in
`~/.config/reachy_mini_openai_chat_open/.env`, readable only by your user) and
the app connects immediately — no files to edit, and it survives restarts.

**Alternative: set it yourself** (e.g. for headless/scripted installs). The app
looks for a `.env` file in, in order:

1. `$REACHY_MINI_OPENAI_ENV` (an explicit path you set)
2. the current working directory
3. `~/.config/reachy_mini_openai_chat_open/.env`  ← recommended on the robot
4. next to the installed app

```bash
mkdir -p ~/.config/reachy_mini_openai_chat_open
cp .env.example ~/.config/reachy_mini_openai_chat_open/.env
nano ~/.config/reachy_mini_openai_chat_open/.env   # paste your OPENAI_API_KEY
```

You can also just export it in the environment the daemon runs in:

```bash
export OPENAI_API_KEY=sk-...
```

## 4. Run it

Start it from the dashboard, or from the REST API:

```bash
curl -X POST http://localhost:8000/api/apps/start-app/openai_chat_open
# stop:
curl -X POST http://localhost:8000/api/apps/stop-current-app
```

Or run the module directly for debugging (with the daemon running):

```bash
python -m reachy_mini_openai_chat_open.main
```

Then just **talk to it**. Try:

- "Hey Reachy, what can you see right now?"
- "Do a little dance!"
- "Follow my face."
- "How are you feeling today?"

### Settings page (in the dashboard)

Once the app is running, it serves a settings page at **`http://localhost:8042`**
(on the Lite; on Wireless it's `http://reachy-mini.local:8042`). The Reachy Mini
dashboard also shows a settings icon that opens it. From there you can, without
touching a config file:

- enter or replace your **OpenAI API key** (stored on the robot; the page never displays the full key),
- toggle behaviours **live** (face tracking, ambient motion, half-duplex),
- change **voice and personality** (system prompt, greeting, model) — saving briefly reconnects the OpenAI session to apply them. The system prompt and greeting are also kept for next time (see below). Conversations are always in English,
- see a **live camera view** of what the robot sees (a few frames a second, streamed only to your browser; pause it with the ⏸ button),
- **test gestures** with one click (nod, dance, wiggle, …), and
- watch the **live transcript** of the conversation (in-memory only; it is gone when the app stops).

Everything on this page maps to the same settings below, so the `.env` file is
still the way to set defaults; the panel is for tweaking on the fly.

**What sticks between runs.** The **system prompt**, **greeting** and **model**
you save on the page are written to
`~/.config/reachy_mini_openai_chat_open/settings.json` and reloaded next time
the app starts — so a personality you like is still there tomorrow. Because they
were chosen deliberately and later, they take precedence over
`REACHY_INSTRUCTIONS` / `REACHY_GREETING` / `OPENAI_REALTIME_MODEL` in your
`.env`; delete that file to go back to the env/built-in defaults. Everything
else on the page (voice, toggles, volume) applies for the current run only.

### Running from your computer or your phone

You can start and control the app from the **Reachy Mini desktop app** or the
**Reachy Mini iPhone app** — both are front-ends to the daemon, and the app that
runs is the same. A few things worth knowing, especially on the **Lite**:

- **The daemon runs on your host computer.** The Lite has no onboard compute, so that computer must be on and running the daemon, with your phone on the same Wi-Fi. You sign into Hugging Face in the phone app to reach your apps.
- **The conversation happens on the robot.** The app uses the robot's own microphone and camera and speaks through the robot's speaker — the phone/desktop is only the "start / stop / settings" remote. You talk to the robot, not the phone.
- **The phone's WebRTC session doesn't get in the way.** When the phone connects it opens a WebRTC media session to the daemon; the daemon is designed so a remote viewer (your phone) and a local app (this one) can read the camera and mic at the same time without conflict.
- **Echo cancellation is built in.** The robot's mic has hardware acoustic echo cancellation "so Reachy Mini doesn't end up talking to itself," so you can usually leave `REACHY_HALF_DUPLEX=false` and only enable it if you still hear it interrupting itself.
- **Settings panel on mobile:** if the phone app doesn't embed the `:8042` page, open `http://<host-computer-ip>:8042` in your phone's browser while on the same Wi-Fi.

## 5. Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI key |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | Realtime model (overridden by a value saved on the settings page) |
| `OPENAI_REALTIME_VOICE` | `marin` | Voice (marin, cedar, alloy, echo, shimmer, …) |
| `REACHY_INSTRUCTIONS` | built-in | System prompt / personality (overridden by a value saved on the settings page) |
| `REACHY_GREETING` | built-in | Spoken word for word on start, with the mic held shut until it finishes (same override) |
| `REACHY_GREET_ON_START` | `true` | Greet when the app launches |
| `REACHY_ENABLE_CAMERA` | `true` | Enable camera vision + face tracking |
| `REACHY_BLUR_FACES` | `true` | Pixelate faces before a camera frame is sent to OpenAI |
| `REACHY_FACE_TRACKING` | `false` | Start with face-follow enabled |
| `REACHY_AMBIENT` | `true` | Ambient breathing / idle motion |
| `REACHY_HALF_DUPLEX` | `false` | Mute mic while the robot talks (fixes echo/self-interruption) |
| `REACHY_SPEAKER_LEAD_MS` | `250` | Speech kept buffered in the daemon while the robot talks — the most that can play on after you interrupt. Raise it if the voice stutters |

**Personality:** the quickest way to change who the robot is, is to set
`REACHY_INSTRUCTIONS`. It's just the system prompt for the Realtime model.

## 6. Privacy

- **No recording.** Conversations are not saved to disk, not uploaded, and not
  analysed. The settings page's live transcript is an in-memory ring buffer
  that disappears when the app stops.
- **What leaves the robot:** your microphone audio and (only when the model
  calls the `look` tool) a single camera frame go to the **OpenAI Realtime
  API** under your own key — that's the whole point of the app. Faces in
  camera frames are pixelated locally first (`REACHY_BLUR_FACES=true`).
- **Your API key** stays on the robot in a `0600` `.env` file and is never
  shown back in full by the settings page.

## 7. How it works

```
robot mic ─16kHz float─▶ resample 24kHz PCM16 ─▶ OpenAI Realtime ─▶ PCM16 ─▶ resample ─▶ robot speaker
                                                     │
                                          function calls (tools)
                                                     ▼
                         MotionController (50 Hz): ambient + talking wobble + gestures + face-track
                                                     ▲
                                        camera frame ─(input_image)─┘  ("what do you see?")
```

- `realtime.py` — OpenAI Realtime WebSocket client (audio + tool dispatch + vision).
- `motion.py` — 50 Hz motion controller blending ambient, speech-reactive, gesture and tracking layers.
- `audio.py` — resampling and a barge-in-capable playback queue with a live "talking energy" meter.
- `vision.py` — camera JPEG capture (for the model) and a local OpenCV face tracker.
- `tools.py` — the `express`, `look`, and `set_face_tracking` tools the model can call.
- `transcript.py` — the in-memory live transcript shown in the settings page.
- `webui.py` + `static/` — the dashboard settings page (FastAPI routes on `self.settings_app`, served at `:8042`).
- `config.py` — env / `.env` configuration, plus the settings-page values persisted to `settings.json`.
- `main.py` — the `ReachyMiniApp` that starts and supervises everything.

## 8. Publishing your own copy

**With the CLI (easiest — ships with `reachy-mini`):**

```bash
hf auth login                      # token with Write access
reachy-mini-app-assistant publish /path/to/reachy_mini_openai_chat_open
```

When prompted, choose **private** or **public**.

**In the browser (no terminal):** create a **New → Space**, SDK **Static**, the
visibility you want, and upload the folder *contents* to the Space root.

The `tags: [reachy_mini_python_app]` line in this README's frontmatter is what
makes it show up as an installable Reachy Mini app — keep it. Only the
`.env.example` placeholder should ever be in the Space; **never** upload a real
`.env` with your key.

## Notes & tips

- **Echo / the robot interrupts itself:** the robot mic has hardware echo cancellation, so usually you need nothing. If it still happens, set `REACHY_HALF_DUPLEX=true`.
- **No camera vision:** ensure `opencv-python` is installed and the media backend has a camera; set `REACHY_ENABLE_CAMERA=true`.
- **Model name changes:** if OpenAI retires the `gpt-realtime` alias, set `OPENAI_REALTIME_MODEL` in your `.env` — no code change needed.
- **Costs:** the Realtime API bills for audio in/out and any image frames sent — usage is on your OpenAI account.
