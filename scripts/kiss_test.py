#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal KISS-over-TCP test client for the 8PSK `modem` executable.

No dependencies beyond the standard library - meant as a quick way to send
or receive a real AX.25-ish frame through the KISS interface, without needing
a full AX.25 stack or external tooling. Frame *content* is arbitrary bytes;
this project's TNC (hdlc.rs) never interprets it.

Usage:
    kiss_test.py send <host> <port> <message>
        Connects, sends one KISS data frame containing <message> (as UTF-8
        bytes), then exits. Use against the modem's KISS port (the same port also carries received frames).

    kiss_test.py listen <host> <port>
        Connects and prints every KISS data frame received, until Ctrl+C.
        Use against the modem's KISS port; frames the modem receives off the air arrive here.

    kiss_test.py loopback <host> <port> <message>
        Sends one frame, then also listens on the same connection - useful
        against a single modem on ONE machine (e.g. bladeRF
        loopback mode) where the same process could plausibly see its own
        frame come back. Against two separate machines, just run `send` on
        one and `listen` on the other instead.

Examples:
    ./kiss_test.py send 192.168.1.10 8001 "hello over the air"
    ./kiss_test.py listen 192.168.1.204 8001
"""

import socket
import sys
import time

FEND = 0xC0
FESC = 0xDB
TFEND = 0xDC
TFESC = 0xDD
DATA_COMMAND = 0x00


def kiss_encode(payload: bytes) -> bytes:
    out = bytearray()
    out.append(FEND)
    out.append(DATA_COMMAND)
    for b in payload:
        if b == FEND:
            out.append(FESC)
            out.append(TFEND)
        elif b == FESC:
            out.append(FESC)
            out.append(TFESC)
        else:
            out.append(b)
    out.append(FEND)
    return bytes(out)


class KissDecoder:
    """Streaming KISS byte decoder - mirrors src/kiss.rs's KissDecoder."""

    def __init__(self):
        self._buf = bytearray()
        self._escaped = False

    def feed(self, data: bytes):
        frames = []
        for b in data:
            if self._escaped:
                self._escaped = False
                if b == TFEND:
                    self._buf.append(FEND)
                elif b == TFESC:
                    self._buf.append(FESC)
                else:
                    self._buf.append(b)  # malformed escape - lenient passthrough
                continue
            if b == FEND:
                if self._buf:
                    if self._buf[0] & 0x0F == DATA_COMMAND:
                        frames.append(bytes(self._buf[1:]))
                    self._buf = bytearray()
            elif b == FESC:
                self._escaped = True
            else:
                self._buf.append(b)
        return frames


def describe_frame(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
        return f"{len(payload)} bytes, utf-8: {text!r}"
    except UnicodeDecodeError:
        return f"{len(payload)} bytes, hex: {payload.hex()}"


def cmd_send(host: str, port: int, message: str):
    payload = message.encode("utf-8")
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(kiss_encode(payload))
        print(f"Sent {len(payload)}-byte frame to {host}:{port}: {message!r}")


def cmd_listen(host: str, port: int, sock=None):
    owns_socket = sock is None
    if sock is None:
        sock = socket.create_connection((host, port), timeout=5)
    sock.settimeout(1.0)
    print(f"Listening for KISS frames on {host}:{port}. Ctrl+C to stop.")
    decoder = KissDecoder()
    count = 0
    try:
        while True:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                print("Connection closed by remote end.")
                break
            for frame in decoder.feed(data):
                count += 1
                print(f"[{count}] {describe_frame(frame)}")
    except KeyboardInterrupt:
        print(f"\nStopped. Received {count} frame(s).")
    finally:
        if owns_socket:
            sock.close()


def cmd_loopback(host: str, port: int, message: str):
    payload = message.encode("utf-8")
    sock = socket.create_connection((host, port), timeout=5)
    sock.sendall(kiss_encode(payload))
    print(f"Sent {len(payload)}-byte frame: {message!r}")
    time.sleep(0.2)
    cmd_listen(host, port, sock=sock)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    if action == "send" and len(sys.argv) >= 5:
        cmd_send(sys.argv[2], int(sys.argv[3]), " ".join(sys.argv[4:]))
    elif action == "listen" and len(sys.argv) >= 4:
        cmd_listen(sys.argv[2], int(sys.argv[3]))
    elif action == "loopback" and len(sys.argv) >= 5:
        cmd_loopback(sys.argv[2], int(sys.argv[3]), " ".join(sys.argv[4:]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
