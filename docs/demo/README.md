# Project demos

## Terminal walkthrough

https://github.com/user-attachments/assets/2ab5fb82-63d8-4ec7-8c25-6c92ca5e4261

[Watch/download the MP4](terminal-walkthrough.mp4) · [Replayable terminal session](terminal-walkthrough.cast) · [Walkthrough source](../../examples/terminal_walkthrough.py)

The walkthrough executes real project commands on an RTX 5090:

1. Check PyTorch, CUDA availability and GPU identity.
2. Batch four dense requests into one forward; reject a fifth request at admission.
3. Group sequence lengths 5 and 7 into bucket 8, while length 31 uses bucket 32.
4. Check HTTP readiness and send an inference request to a local GPU service.
5. Run a five-second independent-process load test and inspect measured outcomes.

[Recorded dense-request output](../../results/demo.json) and [load-test output](../../results/demo-benchmark.json) preserve the actual results shown in the video, including request shapes, counts, latency histograms, resource observations and correctness checks.

The presentation is scripted; command output is captured from a real Windows ConPTY session. The MP4 renders the captured terminal characters and their original timing at 1280 × 800, 12 fps. It is silent. No command results are synthesized. The short load test is a functional demonstration, not a maximum-throughput measurement.

Run the same walkthrough after installing the project:

```bash
python examples/terminal_walkthrough.py --device cuda
```

For individual examples:

```bash
python examples/quick_demo.py --device cuda
python examples/use_cases.py --case sequence --device cuda
python examples/use_cases.py --case service --device cuda
```

Each accepts `--device cpu`. The sequence and HTTP examples compare outputs with direct model execution.

To make a new terminal recording on Windows, install the optional capture dependencies and FFmpeg:

```bash
python -m pip install pywinpty pyte pillow
python scripts/record_terminal.py --device cuda --ffmpeg /path/to/ffmpeg.exe
```

Use `--replay` to encode the existing `.cast` without running inference again. An asciinema-compatible player can also replay the terminal log.

## Request-flow animation

![One burst becomes one GPU batch and four CPU results](runtime-demo.gif)

[Recorded request data](recording.json) accompanies this explanatory animation. Four `[1, 256]` inputs produce one `[4, 256]` dense-model forward and four independent `[1, 32]` CPU results. A fifth request exceeds the four-request admission limit. All returned rows match direct-model output.

The capture gates a controlled burst to make admission and batch membership deterministic; ordinary application callers use `submit` without this recording gate. The animation expands the recorded operations into schematic stages; their display durations are not measured GPU timings.

Recreate the animation with Pillow installed:

```bash
python examples/quick_demo.py --device cuda --output docs/demo/recording.json \
    --render docs/demo/runtime-demo.gif
```

`SHA256SUMS` records the shipped recordings and media files.
