# Imaging Source DFK 72AUC02 – Minimal Setup (Jetson Orin Nano / Linux)

This guide captures only what actually worked in our session to get `/dev/videoX` and capture images.

## 1) Put the camera into UVC mode (critical)

- Build the firmware update tool and flash the UVC firmware 3012.
- Source and tooling from The Imaging Source: `tcam-firmware-update` and firmware file `dfk72uc02_3012.euvc`.
  - Reference: `tcam-firmware-update` and `tiscamera` repositories by The Imaging Source [TheImagingSource/tcam-firmware-update](https://github.com/TheImagingSource) and [TheImagingSource/tiscamera](https://github.com/TheImagingSource).

Commands:
```bash
sudo apt update
sudo apt install -y git cmake build-essential libzip-dev libusb-1.0-0-dev

mkdir -p /tmp/tiscamera_install && cd /tmp/tiscamera_install

git clone https://github.com/TheImagingSource/tcam-firmware-update.git
cd tcam-firmware-update && mkdir -p build && cd build
cmake .. && make -j$(nproc)

# List devices → note SERIAL (e.g., 27420574)
./bin/tcam-firmware-update -l
# Inspect current firmware/mode
./bin/tcam-firmware-update -i -d SERIAL

# Download UVC firmware
cd /tmp/tiscamera_install
wget -O dfk72uc02_3012.euvc \
  https://raw.githubusercontent.com/TheImagingSource/tcam-firmware-update/master/firmware/usb2/dfk72uc02_3012.euvc

# Flash UVC firmware (enables standard V4L2 /dev/videoX)
cd /tmp/tiscamera_install/tcam-firmware-update/build
sudo ./bin/tcam-firmware-update -u -f ../../dfk72uc02_3012.euvc -d SERIAL
```
Then unplug/replug the camera (or reboot).

Verify nodes:
```bash
ls -l /dev/video*
```
You should see `/dev/video0` (and possibly `/dev/video1`).

## 2) Install runtime tools used

```bash
sudo apt install -y v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly}
```

## 3) Test capture (working pipelines/commands)

- Single snapshot to JPEG via GStreamer (works):
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! jpegenc ! filesink location=frame.jpg
```

- Live preview:
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

- Control exposure/gain/brightness with V4L2:
```bash
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=120
v4l2-ctl --device=/dev/video0 --set-ctrl=gain=20
v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_time_absolute=10000
```

Notes:
- If you get "Device busy", ensure no previous gst-launch process is running:
```bash
lsof /dev/video0
pkill -f gst-launch
```

## 4) Optional: Official samples (reference only)
- Official Linux samples and TIS SDK: [TheImagingSource/Linux-tiscamera-Programming-Samples](https://github.com/TheImagingSource/Linux-tiscamera-Programming-Samples), [TheImagingSource/tiscamera](https://github.com/TheImagingSource/tiscamera). We did not rely on Python GI bindings due to extra dependencies.

That’s it. These steps are the minimal, reproducible path we verified end-to-end: flash UVC firmware → verify `/dev/videoX` → capture with GStreamer → adjust parameters with `v4l2-ctl`. 

## 5) End-to-End Workflow We Used (What made it work)

### What we changed/fixed
- Freed the camera when it was busy (stale `gst-launch-1.0` holding `/dev/video0`).
- Ensured GStreamer core and video plugins were installed (`base`, `good`, `bad`, `libav`, `gl`).
- Used system Python (`/usr/bin/python3`) so `gi` (GObject Introspection) and GStreamer bindings are available to the app.
- Selected GREY (GRAY8) modes that the camera supports and set sane exposure values to allow real frame rates.
- Used `v4l2-ctl` to turn off vendor trigger and set exposure for continuous streaming.

### Continuous preview (GStreamer)
- Kill anything using the camera:
```bash
pids=$(fuser -v /dev/video0 2>/dev/null | awk 'NR>1{print $2}'); for p in $pids; do kill -9 "$p" || true; done
```
- Disable triggers and set exposure (example: 10ms):
```bash
v4l2-ctl -d /dev/video0 -c trigger=0 || true
v4l2-ctl -d /dev/video0 -c software_trigger=0 || true
v4l2-ctl -d /dev/video0 -c trigger_global_reset_shutter=0 || true
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=10000 || true
```
- Live view at GRAY8 1280x720@30:
```bash
gst-launch-1.0 -v v4l2src device=/dev/video0 io-mode=2 \
  ! video/x-raw,format=GRAY8,width=1280,height=720,framerate=30/1 \
  ! videoconvert ! autovideosink sync=false -e
```
- If preview window appears but stutters, reduce exposure further (e.g., 3000–5000) or switch to 640x480@60.

### Single image capture (no code changes)
- JPEG from current GREY mode:
```bash
gst-launch-1.0 -e v4l2src device=/dev/video0 num-buffers=1 \
  ! video/x-raw,format=GRAY8,width=1280,height=720 \
  ! jpegenc ! filesink location=frame.jpg
```
- Raw dump via V4L2 (fastest):
```bash
v4l2-ctl -d /dev/video0 --stream-mmap=3 --stream-count=1 --stream-to=frame.raw
```

### Optional: Use The Imaging Source SDK (tiscamera)
- tiscamera provides `tcambin`, `tcam-ctrl`, and GUI `tcam-capture` with robust device properties and debayer.
- Source and instructions: [tiscamera on GitHub](https://github.com/TheImagingSource/tiscamera).
- Build (summary):
```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl \
  libgirepository1.0-dev python3-gi

cd /tmp
git clone https://github.com/TheImagingSource/tiscamera.git
cd tiscamera
sudo ./scripts/dependency-manager install
mkdir build && cd build
cmake -DTCAM_BUILD_ARAVIS=OFF -DTCAM_BUILD_V4L2=ON -DTCAM_BUILD_TOOLS=ON ..
make -j$(nproc)
sudo make install
sudo ldconfig
```
- After install:
```bash
tcam-ctrl --list-devices           # list
tcam-ctrl --list                   # properties (single device) or --device SERIAL
tcam-capture                       # GUI preview and capture
```

### Helper script for TIS (added to repo)
- Script: `tis_cam.sh` in the project root.
- Common commands:
```bash
./tis_cam.sh list
./tis_cam.sh props
./tis_cam.sh preview             # auto-detect single device, 1280x720@30 GRAY8
./tis_cam.sh capture             # single JPEG frame (frame.jpg)
./tis_cam.sh set SERIAL ExposureAuto=false ExposureTime=10000 TriggerMode=false
```


