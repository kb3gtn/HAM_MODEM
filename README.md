# ham_modem

A software modem for amateur radio on a bladeRF SDR: BPSK / QPSK / 8PSK (Gray
mapped, RRC pulse shaping), K=7 rate-1/2 convolutional FEC with a soft-decision
Viterbi decoder, and AX.25/HDLC framing exposed through a KISS TCP port. The
default mode is 8PSK at 16 ksym/s: 48 kbps on air, 24 kbps of data, in about
21.6 kHz of bandwidth. Control and telemetry go over MQTT.

## Build

    cargo build --release
    cargo test --release

The `modem` binary is the main program (`target/release/modem`); the other
binaries in `src/bin` are calibration and loopback-capture tools. See the
header comment in `src/bin/modem.rs` for all command-line options.

## Tools

- `scripts/modem_gui.py` - PySide6 GUI to start/stop the modem and watch
  telemetry (needs `PySide6` and `paho-mqtt`).
- `scripts/kiss_test.py` - minimal KISS-over-TCP send/receive test client.

## IP over the link

Point `tncattach` at the modem's KISS port to carry IP traffic, then test with
`iperf`.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
