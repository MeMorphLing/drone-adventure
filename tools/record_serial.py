"""Record training audio from the node's own microphone.

Flash the node with `#define RECORD_MODE` uncommented in node/src/config.h, then:
    python tools/record_serial.py --port /dev/ttyACM0 --seconds 30 --out training/data/drone/toy_quad_01.wav
(Windows: --port COM5). Record each class in the place you will demo, at several distances.
"""
import argparse
import wave

import serial  # pip install pyserial

SYNC = b"\xA5\x5A"
FRAME_SAMPLES = 512
FRAME_BYTES = 2 + FRAME_SAMPLES * 2
RATE = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    want = int(args.seconds * RATE)
    pcm = bytearray()
    buf = bytearray()
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        ser.reset_input_buffer()
        print(f"recording {args.seconds:.0f} s from {args.port} ...")
        while len(pcm) < want * 2:
            buf += ser.read(4096)
            while True:
                i = buf.find(SYNC)
                if i < 0 or len(buf) - i < FRAME_BYTES:
                    if i > 0:
                        del buf[:i]
                    break
                pcm += buf[i + 2:i + FRAME_BYTES]
                del buf[:i + FRAME_BYTES]
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(pcm[:want * 2]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
