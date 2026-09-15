# Running Reachy Mini OpenAI Chat (Open) on a new computer

This guide takes you from a fresh computer to talking with the robot. It covers
the **Reachy Mini Lite** (the robot is plugged into your computer, which runs
the daemon). If you have a **Wireless** Reachy Mini, the app runs on the robot
itself; see "Install on the robot" in [README.md](README.md) instead.

No robot? Follow steps 1 to 5, then jump to
[Running without a robot (simulator)](#10-running-without-a-robot-simulator).

Only the `reachy-mini` Python package is needed. The Reachy Mini desktop app is
optional: it is a GUI that launches the same daemon for you.

## 1. Prerequisites

Install these on the new computer:

- **Git** — https://git-scm.com/downloads
- **Python 3.10 or newer** — https://www.python.org/downloads/
  (on Windows, tick "Add python.exe to PATH" in the installer)
- **uv** (optional, but faster than pip):

  ```bash
  pip install uv
  ```

You also need an **OpenAI API key** with Realtime API access.

## 2. Clone the repository

```bash
git clone https://github.com/bethqzak/reachy_mini_openai_chat_open.git
cd reachy_mini_openai_chat_open
```

## 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Every later step assumes this
environment is active.

## 4. Install the app

From the repository folder:

```bash
pip install -e .
# or, faster:
uv pip install -e .
```

This installs the app in editable mode together with its dependencies:
`reachy-mini` (which includes the daemon), `numpy`, `scipy`, `websockets`,
`opencv-python` and `python-dotenv`. Editable mode means a later `git pull`
takes effect without reinstalling.

## 5. Add your OpenAI key

Pick one of the two options.

**Option A — use the settings page (easiest).** Skip this step for now. Once the
app is running (step 7), open http://localhost:8042 and paste your key into the
**OpenAI API key** box. It is saved to
`~/.config/reachy_mini_openai_chat_open/.env` and survives restarts.

**Option B — create a `.env` file.**

Windows:

```powershell
copy .env.example .env
notepad .env
```

macOS / Linux:

```bash
cp .env.example .env
nano .env
```

Set the line `OPENAI_API_KEY=sk-...` to your real key and save. The other
settings in the file are optional.

> Never commit `.env`. It is already listed in `.gitignore`.

## 6. Start the daemon

Plug the robot in, then in a terminal with the virtual environment active:

```bash
reachy-mini-daemon
```

Leave this terminal open. The daemon must be running before the app starts, or
the app cannot find the robot. The Reachy Mini dashboard is now at
http://localhost:8000.

## 7. Run the app

Open a **second** terminal, activate the virtual environment again (step 3),
then either:

**Run it directly** (best for development, logs appear in the terminal):

```bash
python -m reachy_mini_openai_chat_open.main
```

**Or start it from the dashboard:** open http://localhost:8000, find
*OpenAI Chat (Open)* in the app list, and press start. The editable install
registers the app with the daemon, so it appears automatically.

**Or start it from the REST API:**

```bash
curl -X POST http://localhost:8000/api/apps/start-app/openai_chat_open
# stop:
curl -X POST http://localhost:8000/api/apps/stop-current-app
```

Then talk to the robot. Try "Hey Reachy, what can you see?" or "Do a little
dance!"

## 8. Settings page

While the app runs, open http://localhost:8042 to:

- enter or replace the OpenAI API key,
- change voice, personality (system prompt), greeting and model,
- toggle face tracking, ambient motion and half-duplex live,
- test gestures,
- watch the live transcript (in memory only).


The system prompt, greeting and model you save there are stored in
`~/.config/reachy_mini_openai_chat_open/settings.json` and reloaded next time.
On Windows that folder is `C:\Users\<you>\.config\reachy_mini_openai_chat_open\`.

## 9. Keeping it up to date

```bash
cd reachy_mini_openai_chat_open
git pull
```

Because the app is installed in editable mode, no reinstall is needed unless
`pyproject.toml` changed. If it did, run `pip install -e .` again.

## 10. Running without a robot (simulator)

The daemon can simulate the robot, so you can develop and test the app with no
hardware attached. The app itself needs no changes: it talks to the daemon the
same way. Do steps 1 to 5 first, then follow this section instead of steps 6
and 7.

### Option A — MuJoCo simulator with a 3D view (recommended)

Install the MuJoCo extra once (it is not part of the app's dependencies):

```bash
pip install "reachy-mini[mujoco]"
```

Start the daemon in simulation mode:

```bash
reachy-mini-daemon --sim
```

A MuJoCo window opens showing the robot. Head, body and antenna movements are
played in it, so you can see gestures, ambient motion and the talking wobble.
Leave this terminal open. The dashboard is at http://localhost:8000 as usual.

Add `--headless` if you do not want the 3D window (for example on a server).

What the app gets in this mode:

- **Camera:** frames from the simulated camera, i.e. a render of the MuJoCo
  scene. "What do you see?" will describe the empty simulated room. Face
  tracking finds nothing.
- **Microphone and speaker:** your computer's default mic and speakers, with
  software echo cancellation. If the robot keeps hearing itself, use headphones
  or set `REACHY_HALF_DUPLEX=true` in `.env`.

### Option B — mockup simulator (no MuJoCo, no 3D view)

```bash
reachy-mini-daemon --mockup-sim
```

This needs nothing extra. Motion commands are accepted but there is nothing to
look at. The camera is your computer's webcam, so vision and face tracking work
on what your webcam sees. Audio is your computer's mic and speakers, as above.

### Run the app against the simulator

In a second terminal with the virtual environment active:

```bash
python -m reachy_mini_openai_chat_open.main
```

Or start it from the dashboard at http://localhost:8000. The settings page is
at http://localhost:8042 as usual. Speak into your computer's microphone.

## Troubleshooting

| Problem | What to check |
|---|---|
| `reachy-mini-daemon` or `python` not found | The virtual environment is not active, or Python is not on PATH. Redo step 3. |
| App cannot connect to the robot | Start the daemon (step 6) first and confirm http://localhost:8000 loads. |
| "OpenAI connection failed" | The key is missing or has no Realtime access. Set it on http://localhost:8042 or in `.env`. |
| No camera vision | Make sure `opencv-python` installed (step 4) and `REACHY_ENABLE_CAMERA=true` in `.env`. |
| Robot interrupts itself / echo | Set `REACHY_HALF_DUPLEX=true` in `.env` or toggle it on the settings page. |
| Voice stutters | Raise `REACHY_SPEAKER_LEAD_MS` (default 250) in `.env`. |
| Personality from the old computer is missing | Copy `settings.json` from `~/.config/reachy_mini_openai_chat_open/` on the old machine to the same path on the new one. |
| `--sim` fails with "No module named mujoco" | Run `pip install "reachy-mini[mujoco]"` in the virtual environment (section 10). |
| No sound or mic in the simulator | The daemon uses your computer's default audio devices. Check the OS sound settings, then restart the daemon. |

For a full list of settings, see the **Configuration** table in
[README.md](README.md) or the comments in [.env.example](.env.example).
