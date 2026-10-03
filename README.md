# ham_modem

A software-defined packet radio modem for amateur radio, running on a
[bladeRF](https://www.nuand.com/) SDR. It carries AX.25 packets over the air
at up to 48 kbps using phase-shift keying, and presents them to the host as a
standard KISS TNC over TCP, so it works with existing packet software
(`tncattach`, Linux AX.25, etc.).

## Features

- **Modulation:** BPSK, QPSK or 8PSK (Gray mapped), root-raised-cosine pulse
  shaping (0.35 rolloff), selectable symbol rate (2 ksym/s to 1 Msym/s; default
  16 ksym/s, which is 48 kbps on air for 8PSK in about 21.6 kHz).
- **Scrambling:** G3RUH self-synchronizing scrambler (x^17 + x^12 + 1).
- **FEC:** K=7, rate-1/2 convolutional code with a soft-decision Viterbi
  decoder (24 kbps of data at the default 8PSK settings, before HDLC overhead).
- **Framing:** AX.25 HDLC (flags, bit stuffing, FCS) with a TCP KISS interface.
  Frame contents are passed through as opaque bytes.
- **Receiver:** automatic frequency acquisition, carrier and timing recovery,
  and phase-ambiguity resolution.
- **Control and telemetry:** MQTT, with an optional PySide6 GUI showing
  constellation, SNR, frequency offset and BER.
- **Built-in test modes:** a BERT (PRBS) source for link testing and an
  internal RF loopback for single-radio self-test.

## Requirements

- A bladeRF with libbladeRF installed
- Rust (stable) to build
- An MQTT broker, e.g. `mosquitto`, for control and telemetry
- Optional, for the GUI: Python 3 with `PySide6` (including QtCharts) and
  `paho-mqtt`
- Optional, for IP over the link: `tncattach` and `iperf`

### Installing dependencies

Arch / CachyOS:

    sudo pacman -S --needed rust bladerf mosquitto pyside6 qt6-charts python-paho-mqtt iperf
    yay -S tncattach-bin    # AUR; or build from https://github.com/markqvist/tncattach
    sudo systemctl enable --now mosquitto

Debian / Ubuntu:

    sudo apt install build-essential pkg-config libbladerf-dev bladerf mosquitto python3-pyside6.qtcharts python3-paho-mqtt iperf
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust
    sudo systemctl enable --now mosquitto

Or install the Python parts with pip: `pip install PySide6 paho-mqtt`.

Check that the radio is visible with `bladeRF-cli -p`. If it needs root, add
your user to the `plugdev` group (or install the bladeRF udev rules).

Note: mosquitto 2.x only accepts local connections by default, which is what
you want when the GUI and modem run on the same machine. The default broker
address used by both is `127.0.0.1:1883`.

## Build

    cargo build --release
    cargo test --release

## Running

Start a broker if you don't have one running (`mosquitto`), then run the
modem:

    target/release/modem --tx-freq 439500000 --rx-freq 439500000

Common options (see the header of `src/bin/modem.rs` for the full list):

| Option | Meaning | Default |
|---|---|---|
| `--modulation` | `bpsk`, `qpsk` or `8psk` | `8psk` |
| `--symbol-rate` | symbol rate in Hz | `16000` |
| `--tx-gain` / `--rx-gain` | gain in dB | `-23` / `50` |
| `--source` | `bert` (test pattern) or `data` (KISS packets) | `bert` |
| `--kiss-port` | KISS TCP port | `8001` |
| `--loopback rfic-bist` | single-radio self-test, no antenna needed | off |

For a self-test with one radio and no antenna:

    target/release/modem --loopback rfic-bist --tx-gain -23 --rx-gain 0

**This transmits real RF.** Make sure you are licensed and authorized for the
chosen frequency, and that your antenna or attenuator is connected safely.

### GUI

    python3 scripts/modem_gui.py [broker_host] [broker_port] [base_topic]

The GUI can start and stop the modem process, change settings live, switch the
TX source between BERT and data, and plot receiver telemetry.

### Sending packets

With the TX source set to `data`, any KISS TCP client can connect to the
modem's KISS port (8001 by default). Frames you send go out on the air, and
frames received off the air come back on the same connection.

- `scripts/kiss_test.py send 127.0.0.1 8001 "hello"` sends one frame.
- `scripts/kiss_test.py listen 127.0.0.1 8001` prints received frames.
- Attach `tncattach` to the KISS port to carry IP traffic over the link, then
  test it with `ping` and `iperf`.

## Layout

- `src/` - the modem library (DSP chain, FEC, HDLC, KISS) and `src/bin/modem.rs`,
  the main program. The other binaries in `src/bin` are calibration and
  loopback-capture tools.
- `scripts/` - the GUI and a KISS test client.
- `tests/` - loopback integration tests.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
