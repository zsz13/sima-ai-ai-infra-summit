# Running the demo

Every command is labelled by where it runs:

| Label | Where |
|---|---|
| `[HOST]` | a Terminal on the MacBook |
| `[SDK]` | inside the Palette Neat SDK container (`sima-cli sdk neat`) |
| `[DEVKIT]` | on the Modalix DevKit (`ssh sima@<ip>` or `sima-cli serial`) |

---

## Prerequisites

**Hardware**

- Modalix MLSoC DevKit 3.0, powered on (it takes a couple of minutes to boot)
- USB cable from the kit's **UART** connector to the Mac (serial console)
- **USB-to-Ethernet adapter plugged directly into a USB-C port on the Mac.**
  Not through a hub — the Game Day Guide is explicit about this, and a hub is the
  most common cause of a dead link.
- Ethernet cable from the kit's RJ45 to that adapter
- Mac on Wi-Fi (the DevKit reaches the internet through it)

**Software**

- Docker running, SDK container up: `[HOST] sima-cli sdk start`
- `[HOST] brew install ffmpeg` and `uv`
- A SiMa developer-portal account for the Model Zoo

**Never commit or screenshot the VS Code Web token.** The URL has the form
`https://<ip>:<port>/?tkn=<token>&folder=/workspace`; the token is a credential.

---

## One-time setup

### 1. Put the Mac and DevKit on the same network, with internet for the board

This is the Internet Sharing path from Game Day Guide Appendix 2. It gives the
DevKit both a route to the Mac *and* internet access, which it needs to pull
several GB of models.

1. `[HOST]` System Settings → Network → your USB Ethernet adapter → Details →
   TCP/IP → **Using DHCP**.
2. `[HOST]` System Settings → General → Sharing → **Internet Sharing**. Share
   from **Wi-Fi** to the **USB Ethernet adapter**, then switch Internet Sharing on.
3. `[DEVKIT]` over the serial console (`[HOST] sima-cli serial`, login `sima` /
   `edgeai`):
   ```bash
   sudo nmcli connection down end0-static
   sudo nmcli connection up end0-dhcp
   printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
   ip -4 addr show end0        # expect a 192.168.2.x address
   ping -c 3 google.com
   ```
4. `[HOST]` confirm you can reach it: `ping -c 3 192.168.2.<n>`

### 2. Pair the DevKit with the SDK

```bash
[HOST] sima-cli sdk setup --devkit 192.168.2.<n>
```

This installs the `dk` helper in the container, generates and pushes an SSH key,
and exports `~/workspace` over NFS so the board sees `/workspace`. Answer the
prompts as the Game Day Guide describes; the DevKit password is `edgeai`.

Verify:

```bash
[HOST] ./scripts/smoke-test.sh      # from ~/workspace, all six stages should pass
```

### 3. Get the models

```bash
[HOST] sima-cli login                              # required for the Model Zoo
[HOST] DEVKIT_IP=192.168.2.<n> ./scripts/setup-devkit.sh
```

`setup-devkit.sh` copies the pre-staged Whisper from `/workspace/models` if it is
there (no download), otherwise pulls it, then pulls the vision-language model.

> Model names drift between the docs and the live repositories (`-GPTQ-` vs
> `-Autoround-`). If a pull fails: `[DEVKIT] llima search vl` and pass the real
> name as `VLM_MODEL=...`.

---

## Starting the demo

```bash
[HOST] cd ~/workspace/foreman
[HOST] DEVKIT_IP=192.168.2.<n> \
       FOREMAN_MODEL=/workspace/models/<detector>.tar.gz \
       ./scripts/demo.sh
```

That single command:

1. checks the container, Insight, the DevKit, SSH, and the model
2. publishes the Mac camera to `rtsp://<mac>:8554/src2`
3. starts the GenAI server on the DevKit (vision-language + Whisper on the MLA)
4. starts the Foreman edge agent on the DevKit (detector on the MLA)
5. starts the console on the Mac and opens the browser

Ctrl-C stops everything it started.

### Camera choice

`CAMERA=1` selects the **Desk View** camera, which points down at the bench — a
better inspection angle than the front-facing one.

```bash
[HOST] CAMERA=1 DEVKIT_IP=... FOREMAN_MODEL=... ./scripts/demo.sh
```

List cameras: `[HOST] ffmpeg -f avfoundation -list_devices true -i ""`

---

## What you should see

| Where | URL | What |
|---|---|---|
| **Console** | `http://127.0.0.1:8800` | The product. Verdict, reason, evidence frame, measured latency. |
| **Engineering view** | `https://127.0.0.1:8081/static/viewer.html?src=0` | Neat Insight: live video with detection overlays from the DevKit. |
| **Insight console** | `https://127.0.0.1:9900` | Media sources, MLA metrics, ingest statistics. |

Both Insight URLs use a mkcert certificate, so the browser will warn once.

### The demo, in order

1. The console shows **Modalix connected** with a green dot, and a detector frame
   rate under **See**.
2. Press **State the standard** and say, for example:
   *"Every box must have a label facing up and the lid closed."*
   Whisper transcribes it **on the DevKit** and it appears in the top bar.
3. Hold an item in front of the camera. The console shows **Hold still** with a
   thin amber bar filling as the item settles.
4. The bar completes, the slab reads **Judging**, and a second or two later it
   floods green or red with the verdict and the model's reason in words.
5. The evidence frame on the right is the exact image the model judged.
6. Take the item away and present another. It re-arms automatically.
7. Change the standard by voice mid-demo and present the same item again — the
   verdict changes with no retraining.

---

## If something fails

### The DevKit is unreachable

The single most common cause is the Ethernet adapter.

```bash
[HOST] ifconfig | grep -A3 "^en"          # look for "status: active" and an inet
[HOST] ioreg -p IOUSB -w0 -l | grep "USB Product Name"
```

If no Ethernet adapter appears in that USB list, it is not plugged in, or it is
behind a hub. Plug it **directly into a USB-C port**.

If the adapter is present but has no IP, re-check Internet Sharing (step 1).
The serial console always works regardless: `[HOST] sima-cli serial`.

### `dk: command not found` in the SDK container

Pairing has not been done. `dk` is a shell function written to
`~/.devkit-sync.rc` by `sima-cli sdk setup --devkit`, not a binary — `which dk`
returning nothing is normal even when it works. Re-run step 2, then open a **new**
shell in the container.

### Neat Insight is unhealthy or the viewer is blank

```bash
[HOST] docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 insight-admin status
[HOST] docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 insight-admin restart
```

Then **hard-reload** the viewer tab. Insight caches `drawing.js` aggressively and
a stale copy shows video with no overlays.

Overlays render on port **8081**, not 9900.

### The camera will not start

```bash
[HOST] cat /tmp/foreman-camera.log
```

`avfoundation` only accepts a frame rate the device advertises exactly, and only
delivers `uyvy422`. The script already handles this; if you are running ffmpeg by
hand you need `-framerate 30 -pixel_format uyvy422`.

macOS will ask for camera permission the first time. Grant it to your terminal.

### The console says "Modalix unreachable"

```bash
[HOST]   curl -s http://192.168.2.<n>:8100/health
[DEVKIT] cat /tmp/foreman-edge.log
[DEVKIT] cat /tmp/foreman-genai.log
```

The console is telling the truth — it never shows a stale or invented verdict
when the edge is down.

### Verdicts never appear

The gate has not fired. Check the console's **See** number is non-zero (the
detector is producing frames) and that the item is actually being detected — the
Insight viewer shows the boxes. If the object has no COCO class the detector
recognises, lower `FOREMAN_MIN_CONF` or set the gate to accept any label.

### A judgement takes too long

The detector and the vision-language model share one MLA. Check
`docs/BENCHMARKS.md` for the measured numbers on this hardware. If it is
unusable, switch to a smaller VLM (`LFM2-VL-1.6B`) via `VLM_MODEL=`.

---

## Stopping

```bash
[HOST] ./scripts/stop.sh
```

Stops the host orchestrator, the camera stream, and both DevKit processes. Insight
and the SDK container are left running.

To also stop the Insight media sources:

```bash
[HOST] docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 \
       curl -sk -X POST https://127.0.0.1:9900/api/mediasrc/stop-all
```

## Full restart

```bash
[HOST] ./scripts/stop.sh
[HOST] docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 insight-admin restart
[HOST] DEVKIT_IP=... FOREMAN_MODEL=... ./scripts/demo.sh
```

---

## Fallback: run without a camera

If the camera fails, serve a recorded clip through Insight instead. The demo
still exercises the full DevKit path.

```bash
[HOST] docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'curl -sk -X POST -H "Content-Type: application/json" \
   -d "{\"index\":1,\"file\":\"video03.mp4\",\"transport\":\"rtsp\"}" \
   https://127.0.0.1:9900/api/mediasrc/assign && \
   curl -sk -X POST -H "Content-Type: application/json" -d "{\"index\":1}" \
   https://127.0.0.1:9900/api/mediasrc/start'

[HOST] SKIP_CAMERA=1 SLOT=src1 DEVKIT_IP=... FOREMAN_MODEL=... ./scripts/demo.sh
```

Upload your own clip from the Insight console at `https://127.0.0.1:9900`,
**Media Sources** tab.
