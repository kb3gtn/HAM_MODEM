#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PySide6 GUI for MQTT-based command/control and telemetry of the BPSK/QPSK/8PSK modem
(the `modem` executable).

The GUI can also START, MONITOR and STOP the modem process itself (the
"Modem process" panel): it builds the `modem` command line from the form,
runs it as a child process, shows its state and console output, and stops it
with a clean MQTT "Shutdown" (falling back to terminate/kill). It also shows the
state of a modem started elsewhere, from <base_topic>/modem/status.

Subscribes to <base_topic>/tx/status, <base_topic>/rx/status,
<base_topic>/rx/symbols and <base_topic>/modem/status, and publishes commands
on <base_topic>/modem/control, <base_topic>/tx/control and <base_topic>/rx/control - see those binaries' doc comments for the exact JSON
shapes, which this GUI mirrors. `base_topic` defaults to "psk8" and is editable
live in the connection bar, so one GUI can be pointed at different
modems (give each its own --mqtt-topic); reconnecting re-subscribes
under the new topic.

Link settings: the "Link settings" panel changes the modulation (BPSK, QPSK,
8PSK) and symbol rate (2 ksym/s to 1 Msym/s) of the TX, the RX or both at once.
The modem rebuilds its chain, the link drops, and the receiver re-acquires.
Above 100 ksym/s the modem switches the SDR from 800 kSPS to 4 MSPS (and sets
the analog filter to the signal width) - the radio is restarted for that. The
RX panel's Reacquire button resets the receiver's timing and carrier loops
without changing anything else. The occupied bandwidth is 1.35 x the symbol
rate: at high rates raise the TX/RX freq shift above half of it (the GUI warns).

Layout: a one-line link summary, the transmitter panel, the receiver panel
(Lock / Signal / Errors / Link sections), a live constellation plot, rolling
trend charts (SNR, total frequency offset, pre-FEC BER, input level) and a
command log. A panel shows STALE when its status has not arrived for a few
seconds (process stopped, or no route to the broker).

Constellation: <base_topic>/rx/symbols carries the 256 most recent
carrier-derotated, amplitude-normalized received symbols once a second as
{"i": [...], "q": [...]}. They are plotted against the ideal constellation points. A
tight cluster on each point means a clean link; a ring or smear means
residual phase/frequency error; a blob means no lock. (The whole picture may
be rotated by a multiple of 360/M degrees - the carrier loop cannot tell which
point is which; the decoders resolve that.)

Telemetry worth knowing about:
  * carrier locked   - the carrier loop has locked to data AND been verified
                       against the signal spectrum (not a 2 kHz alias lock).
  * total offset     - estimated TX/RX frequency error (LO retune + residual).
  * SNR / error std  - symbol SNR from the std-dev of the symbol error; only
                       reported while locked, accurate above ~12 dB.
  * pre-FEC BER      - raw channel errors the Viterbi decoder corrected,
                       measured by re-encoding its output. BERT = the PRBS
                       decoder branch, frames = the HDLC-locked branch.
  * post-FEC BER     - PRBS errors after decoding (BERT mode only).
  * rotations        - the carrier loop's (360/M)-degree phase ambiguity as resolved
                       by the BERT / HDLC decoders.

Requires: PySide6 (with QtCharts for the plots - optional), paho-mqtt 1.x
(classic callback API; written against 1.6.1, not the 2.x
CallbackAPIVersion style).

Usage: modem_gui.py [broker_host] [broker_port] [base_topic]
Defaults: 127.0.0.1 1883 psk8
"""

import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt
from PySide6.QtCore import QObject, QPointF, QProcess, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis

    HAVE_CHARTS = True
except ImportError:  # plots are optional
    HAVE_CHARTS = False

STALE_AFTER_S = 3.5
TREND_WINDOW_S = 180.0
BER_FLOOR = 1e-7  # BER trend is plotted as log10, floored here
CONSTELLATION_LIMIT = 1.7

OK_STYLE = "color: #1a7f37; font-weight: bold;"
BAD_STYLE = "color: #c62828; font-weight: bold;"
WARN_STYLE = "color: #b26a00; font-weight: bold;"
NEUTRAL_STYLE = ""


def fmt_ber(value):
    """BER for display. The decoder's moving average decays toward 0 on a clean
    link (to absurd values like 1e-98); anything below 1e-9 is just 'clean'."""
    if value is None:
        return "-"
    return "<1e-9" if value < 1e-9 else f"{value:.2e}"


def fmt_num(value, spec):
    return "-" if value is None else format(value, spec)


MODULATIONS = {"BPSK": ("Bpsk", 2), "QPSK": ("Qpsk", 4), "8PSK": ("Psk8", 8)}  # label -> (wire name, points)
POINTS = {wire: points for wire, points in MODULATIONS.values()}
LABELS = {wire: label for label, (wire, _) in MODULATIONS.items()}
MIN_SYMBOL_RATE = 2000.0
MAX_SYMBOL_RATE = 1000000.0
LOW_PROFILE_MAX_RATE = 100000.0  # up to here the SDR runs at 800 kSPS, above at 4 MSPS
ROLLOFF = 0.35


def rotation_text(step, modulation="Psk8"):
    """The carrier loop's phase ambiguity, in 360/M-degree steps."""
    if step is None:
        return "-"
    return f"{step * 360 // POINTS.get(modulation, 8)} deg"


def parse_number(text, what, log):
    """float() with a log message instead of an exception; None on failure."""
    try:
        return float(text.strip())
    except ValueError:
        log(f"Invalid {what} '{text}' - expected a number")
        return None


class MqttBridge(QObject):
    """Owns the paho-mqtt client (network loop on a background thread via
    loop_start()) and re-emits everything as Qt signals, so the GUI thread
    never touches the client except to publish() (thread-safe in paho once the
    loop is running)."""

    tx_status = Signal(dict)
    rx_status = Signal(dict)
    rx_symbols = Signal(dict)
    modem_status = Signal(dict)
    connection_changed = Signal(bool, str)
    log_message = Signal(str)

    def __init__(self, base_topic: str = "psk8"):
        super().__init__()
        self.base_topic = base_topic
        self._client = None  # recreated per connect: the client ID derives from base_topic

    @property
    def tx_status_topic(self) -> str:
        return f"{self.base_topic}/tx/status"

    @property
    def tx_control_topic(self) -> str:
        return f"{self.base_topic}/tx/control"

    @property
    def rx_status_topic(self) -> str:
        return f"{self.base_topic}/rx/status"

    @property
    def rx_symbols_topic(self) -> str:
        return f"{self.base_topic}/rx/symbols"

    @property
    def rx_control_topic(self) -> str:
        return f"{self.base_topic}/rx/control"

    @property
    def modem_status_topic(self) -> str:
        return f"{self.base_topic}/modem/status"

    @property
    def modem_control_topic(self) -> str:
        return f"{self.base_topic}/modem/control"

    def connect_to(self, host: str, port: int, base_topic: str):
        self.base_topic = base_topic
        # Recreated (not reused) so the client ID always matches the current
        # base topic - two GUIs sharing a broker but different base topics
        # must not collide on client ID.
        self._client = mqtt.Client(client_id=f"{base_topic}_modem_gui")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        try:
            self._client.connect_async(host, port, keepalive=10)
            self._client.loop_start()
            self.log_message.emit(f"Connecting to {host}:{port} (base topic '{base_topic}') ...")
        except OSError as e:
            self.connection_changed.emit(False, str(e))

    def disconnect(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            for topic in (self.tx_status_topic, self.rx_status_topic, self.rx_symbols_topic, self.modem_status_topic):
                client.subscribe(topic, qos=0)
            self.connection_changed.emit(True, "connected")
            self.log_message.emit(f"Connected. Subscribed to {self.base_topic}/tx/status, /rx/status, /rx/symbols, /modem/status.")
        else:
            self.connection_changed.emit(False, f"connect failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connection_changed.emit(False, "disconnected")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.log_message.emit(f"Bad JSON on {msg.topic}: {e}")
            return
        if msg.topic == self.tx_status_topic:
            self.tx_status.emit(payload)
        elif msg.topic == self.rx_status_topic:
            self.rx_status.emit(payload)
        elif msg.topic == self.rx_symbols_topic:
            self.rx_symbols.emit(payload)
        elif msg.topic == self.modem_status_topic:
            self.modem_status.emit(payload)

    def publish(self, topic: str, payload):
        if self._client is None:
            self.log_message.emit(f"(not connected, dropped) -> {topic}: {json.dumps(payload)}")
            return
        text = json.dumps(payload)
        self._client.publish(topic, text, qos=1)
        self.log_message.emit(f"-> {topic}: {text}")

    def publish_tx_control(self, payload):
        self.publish(self.tx_control_topic, payload)

    def publish_rx_control(self, payload):
        self.publish(self.rx_control_topic, payload)

    def publish_modem_control(self, payload):
        self.publish(self.modem_control_topic, payload)

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected()


class StatusLabel(QLabel):
    """A read-only value label that can be coloured good / bad / warning."""

    def set_value(self, text, state=None):
        self.setText(text)
        self.setStyleSheet({True: OK_STYLE, False: BAD_STYLE, "warn": WARN_STYLE}.get(state, NEUTRAL_STYLE))


class SettingRow(QWidget):
    """A line edit + button that publishes a command (Enter also submits)."""

    def __init__(self, placeholder, button_text, on_submit, width=110):
        super().__init__()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setFixedWidth(width)
        self.button = QPushButton(button_text)
        self.button.clicked.connect(on_submit)
        self.edit.returnPressed.connect(on_submit)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit)
        row.addWidget(self.button)
        row.addStretch()
        self.setLayout(row)

    def text(self):
        return self.edit.text()


class TrendChart(QWidget):
    """A small rolling line chart. `add(t, value)` skips None (gap in data)."""

    def __init__(self, title, y_label, y_min=None, y_max=None):
        super().__init__()
        self.points = deque()
        self.fixed_y = (y_min, y_max)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if not HAVE_CHARTS:
            layout.addWidget(QLabel(f"{title}: (QtCharts not installed)"))
            self.setLayout(layout)
            return
        self.series = QLineSeries()
        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.legend().hide()
        self.chart.setTitle(title)
        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat("%.0f")
        self.axis_x.setTickCount(4)  # fewer ticks so the labels aren't elided
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText(y_label)
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        self.axis_y.setRange(y_min if y_min is not None else 0.0, y_max if y_max is not None else 1.0)
        view = QChartView(self.chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setMinimumHeight(140)
        layout.addWidget(view)
        self.setLayout(layout)

    def add(self, t, value):
        if not HAVE_CHARTS or value is None or not math.isfinite(value):
            return
        self.points.append((t, value))
        while self.points and self.points[0][0] < t - TREND_WINDOW_S:
            self.points.popleft()
        self.series.replace([QPointF(x, y) for x, y in self.points])
        self.axis_x.setRange(max(0.0, t - TREND_WINDOW_S), max(TREND_WINDOW_S, t))
        ys = [y for _, y in self.points]
        lo = self.fixed_y[0] if self.fixed_y[0] is not None else min(ys)
        hi = self.fixed_y[1] if self.fixed_y[1] is not None else max(ys)
        pad = (hi - lo) * 0.1 if hi > lo else 1.0
        self.axis_y.setRange(
            lo if self.fixed_y[0] is not None else lo - pad,
            hi if self.fixed_y[1] is not None else hi + pad,
        )


class ConstellationPlot(QWidget):
    """Scatter plot of the latest received symbols over the ideal constellation points."""

    def __init__(self):
        super().__init__()
        self.last_update = None
        self.points = 8
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.caption = QLabel("Constellation: waiting for <base>/rx/symbols ...")
        layout.addWidget(self.caption)
        if not HAVE_CHARTS:
            layout.addWidget(QLabel("(QtCharts not installed)"))
            self.setLayout(layout)
            return

        self.received = QScatterSeries()
        self.received.setMarkerSize(6.0)
        self.received.setColor(QColor(30, 110, 220, 170))
        self.received.setBorderColor(QColor(30, 110, 220, 0))
        self.ideal = QScatterSeries()
        self.ideal.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        self.ideal.setMarkerSize(11.0)
        self.ideal.setColor(QColor(220, 50, 50, 0))
        self.ideal.setBorderColor(QColor(220, 50, 50))
        self.set_modulation("Psk8")
        self.circle = QLineSeries()
        self.circle.setColor(QColor(150, 150, 150, 120))
        for k in range(0, 361, 5):
            a = math.radians(k)
            self.circle.append(math.cos(a), math.sin(a))

        self.chart = QChart()
        for s in (self.circle, self.received, self.ideal):
            self.chart.addSeries(s)
        self.chart.legend().hide()
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        for axis, align, title in ((self.axis_x, Qt.AlignmentFlag.AlignBottom, "I"), (self.axis_y, Qt.AlignmentFlag.AlignLeft, "Q")):
            axis.setRange(-CONSTELLATION_LIMIT, CONSTELLATION_LIMIT)
            axis.setTickCount(5)
            axis.setLabelFormat("%.1f")
            axis.setTitleText(title)
            self.chart.addAxis(axis, align)
        for s in (self.circle, self.received, self.ideal):
            s.attachAxis(self.axis_x)
            s.attachAxis(self.axis_y)
        view = QChartView(self.chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setMinimumSize(300, 260)
        layout.addWidget(view, 1)
        self.setLayout(layout)

    def set_modulation(self, modulation):
        points = POINTS.get(modulation, 8)
        if points == self.points and HAVE_CHARTS and self.ideal.count() == points:
            return
        self.points = points
        if HAVE_CHARTS:
            step = 2 * math.pi / points
            self.ideal.replace([QPointF(math.cos(step * p), math.sin(step * p)) for p in range(points)])

    def update_symbols(self, payload: dict):
        i, q = payload.get("i", []), payload.get("q", [])
        n = min(len(i), len(q))
        self.last_update = time.monotonic()
        if n == 0:
            return
        # Error vector magnitude against the nearest ideal point, as a quick
        # numeric companion to the picture.
        evm = 0.0
        for x, y in zip(i[:n], q[:n]):
            ang = math.atan2(y, x)
            step = 2 * math.pi / self.points
            nearest = round(ang / step) * step
            evm += (x - math.cos(nearest)) ** 2 + (y - math.sin(nearest)) ** 2
        evm = math.sqrt(evm / n)
        self.caption.setText(f"Constellation: last {n} symbols, EVM vs nearest ideal point {evm:.3f}")
        if HAVE_CHARTS:
            self.received.replace([QPointF(x, y) for x, y in zip(i[:n], q[:n])])


# Per-second modem console lines ("[ 12.0s] level -30.7 dBFS | BERT ...") are
# already shown as telemetry; hide them in the log unless "verbose" is ticked.
STATUS_LINE = re.compile(r"^\[\s*[\d.]+s\]\s+level\s")
DEFAULT_MODEM_PATH = str(Path(__file__).resolve().parent.parent / "target" / "release" / "modem")


class ModemPanel(QGroupBox):
    """Start / monitor / stop the `modem` process, and show the state of a
    modem started elsewhere (from <base>/modem/status)."""

    STOP_GRACE_MS = 5000  # after the MQTT Shutdown, before terminate()
    TERMINATE_GRACE_MS = 3000  # after terminate(), before kill()

    def __init__(self, bridge: MqttBridge, get_connection):
        super().__init__("Modem process")
        self.bridge = bridge
        self.get_connection = get_connection  # () -> (host, port, base_topic)
        self.settings = QSettings("psk8-modem", "gui")
        self.last_modem_status = None  # (monotonic time, payload)
        self.stopping = False
        # Identifies the current stop request; the terminate/kill fallback timers
        # carry the token they were armed with and do nothing if it has changed
        # (otherwise a timer from an earlier, successful stop would kill a
        # modem the user has since restarted).
        self._stop_token = 0

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_output)
        self.process.started.connect(self.on_started)
        self.process.finished.connect(self.on_finished)
        self.process.errorOccurred.connect(self.on_error)
        self._partial = ""

        self.exe_edit = QLineEdit(DEFAULT_MODEM_PATH)
        self.tx_freq = QLineEdit("439500000")
        self.rx_freq = QLineEdit("439500000")
        self.tx_gain = QLineEdit("-23")
        self.rx_gain = QLineEdit("50")
        self.tx_shift = QLineEdit("15000")
        self.rx_shift = QLineEdit("15000")
        self.kiss_port = QLineEdit("8001")
        self.symbol_rate = QLineEdit("16000")
        self.modulation = QComboBox()
        self.modulation.addItems(list(MODULATIONS))
        self.modulation.setCurrentText("8PSK")
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("extra modem options, e.g. --carrier-bw 120")
        for w in (self.tx_freq, self.rx_freq):
            w.setFixedWidth(110)
        for w in (self.tx_gain, self.rx_gain, self.tx_shift, self.rx_shift, self.kiss_port, self.symbol_rate):
            w.setFixedWidth(70)
        self.pattern = QComboBox()
        self.pattern.addItems(["pn15", "pn11", "pn23"])
        self.source = QComboBox()
        self.source.addItems(["bert", "data"])
        self.loopback = QComboBox()
        self.loopback.addItems(["none", "rfic-bist"])
        self.loopback.setToolTip("rfic-bist loops TX back into RX inside the radio chip: a single-radio self-test, no cable.\nSets TX gain -23 / RX gain 0 (the loopback adds its own gain).")
        self.loopback.currentTextChanged.connect(self.on_loopback_changed)
        self.tx_check = QCheckBox("TX")
        self.tx_check.setChecked(True)
        self.rx_check = QCheckBox("RX")
        self.rx_check.setChecked(True)
        self.verbose_check = QCheckBox("show per-second console status lines")
        self.close_check = QCheckBox("stop modem when GUI closes")
        self.close_check.setChecked(True)

        self.start_btn = QPushButton("Start modem")
        self.stop_btn = QPushButton("Stop modem")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.state_label = StatusLabel("stopped")
        self.remote_label = StatusLabel("no modem status on MQTT")

        grid = QGridLayout()
        grid.addWidget(QLabel("Executable:"), 0, 0)
        grid.addWidget(self.exe_edit, 0, 1, 1, 9)
        row1 = [("TX freq Hz", self.tx_freq), ("TX gain dB", self.tx_gain), ("TX shift Hz", self.tx_shift), ("Pattern", self.pattern), ("Source", self.source)]
        row2 = [("RX freq Hz", self.rx_freq), ("RX gain dB", self.rx_gain), ("RX shift Hz", self.rx_shift), ("KISS port", self.kiss_port), ("Loopback", self.loopback)]
        row3 = [("Modulation", self.modulation), ("Symbol rate", self.symbol_rate)]
        for r, row in ((1, row1), (2, row2), (3, row3)):
            for c, (label, widget) in enumerate(row):
                grid.addWidget(QLabel(label + ":"), r, 2 * c)
                grid.addWidget(widget, r, 2 * c + 1)
        enables = QHBoxLayout()
        enables.addWidget(QLabel("Run:"))
        enables.addWidget(self.tx_check)
        enables.addWidget(self.rx_check)
        enables.addSpacing(20)
        enables.addWidget(self.extra, 1)
        grid.addLayout(enables, 4, 0, 1, 10)

        controls = QHBoxLayout()
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(QLabel("Process:"))
        controls.addWidget(self.state_label)
        controls.addSpacing(16)
        controls.addWidget(QLabel("On MQTT:"))
        controls.addWidget(self.remote_label, 1)
        options = QHBoxLayout()
        options.addWidget(self.verbose_check)
        options.addWidget(self.close_check)
        options.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addLayout(controls)
        layout.addLayout(options)
        self.setLayout(layout)

        self._fields = {
            "exe": self.exe_edit, "tx_freq": self.tx_freq, "rx_freq": self.rx_freq, "tx_gain": self.tx_gain, "rx_gain": self.rx_gain,
            "tx_shift": self.tx_shift, "rx_shift": self.rx_shift, "kiss_port": self.kiss_port, "symbol_rate": self.symbol_rate, "extra": self.extra,
        }
        self.load_settings()
        self.bridge.modem_status.connect(self.on_modem_status)
        self.refresh_state()

    # ---- settings ----
    def load_settings(self):
        for key, widget in self._fields.items():
            value = self.settings.value(key)
            if value is not None and str(value) != "":
                widget.setText(str(value))
        if not Path(self.exe_edit.text()).is_file():
            self.exe_edit.setText(DEFAULT_MODEM_PATH)  # a stale saved path is worse than the default
        for key, combo in (("pattern", self.pattern), ("source", self.source), ("loopback", self.loopback), ("modulation", self.modulation)):
            value = self.settings.value(key)
            if value is not None:
                i = combo.findText(str(value))
                if i >= 0:
                    combo.blockSignals(True)  # don't let the loopback preset overwrite saved gains
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
        for key, box in (("tx_on", self.tx_check), ("rx_on", self.rx_check), ("verbose", self.verbose_check), ("stop_on_close", self.close_check)):
            value = self.settings.value(key)
            if value is not None:
                box.setChecked(str(value).lower() == "true")

    def save_settings(self):
        for key, widget in self._fields.items():
            self.settings.setValue(key, widget.text())
        for key, combo in (("pattern", self.pattern), ("source", self.source), ("loopback", self.loopback), ("modulation", self.modulation)):
            self.settings.setValue(key, combo.currentText())
        for key, box in (("tx_on", self.tx_check), ("rx_on", self.rx_check), ("verbose", self.verbose_check), ("stop_on_close", self.close_check)):
            self.settings.setValue(key, box.isChecked())

    def on_loopback_changed(self, mode):
        # The radio-chip loopback adds its own gain: these are the settings that work.
        if mode == "rfic-bist":
            self.tx_gain.setText("-23")
            self.rx_gain.setText("0")
        else:
            self.rx_gain.setText("50")

    # ---- process control ----
    def build_args(self):
        log = self.bridge.log_message.emit
        numbers = {}
        for name, widget, kind in (
            ("tx-freq", self.tx_freq, int), ("rx-freq", self.rx_freq, int), ("tx-gain", self.tx_gain, int), ("rx-gain", self.rx_gain, int),
            ("tx-shift", self.tx_shift, float), ("rx-shift", self.rx_shift, float), ("kiss-port", self.kiss_port, int),
        ):
            try:
                numbers[name] = kind(float(widget.text().strip())) if kind is int else float(widget.text().strip())
            except ValueError:
                log(f"Modem: --{name} '{widget.text()}' is not a number")
                return None
        if not (self.tx_check.isChecked() or self.rx_check.isChecked()):
            log("Modem: enable TX and/or RX")
            return None
        try:
            rate = float(self.symbol_rate.text().strip())
        except ValueError:
            log(f"Modem: symbol rate '{self.symbol_rate.text()}' is not a number")
            return None
        if not MIN_SYMBOL_RATE <= rate <= MAX_SYMBOL_RATE:
            log(f"Modem: symbol rate must be {MIN_SYMBOL_RATE:.0f}..{MAX_SYMBOL_RATE:.0f}")
            return None
        host, port, base = self.get_connection()
        args = []
        for name, value in numbers.items():
            args += [f"--{name}", str(value)]
        args += ["--pattern", self.pattern.currentText(), "--source", self.source.currentText()]
        args += ["--modulation", MODULATIONS[self.modulation.currentText()][0].lower(), "--symbol-rate", str(rate)]
        args += ["--mqtt-host", host, "--mqtt-port", str(port), "--mqtt-topic", base]
        if not self.tx_check.isChecked():
            args.append("--no-tx")
        if not self.rx_check.isChecked():
            args.append("--no-rx")
        if self.loopback.currentText() != "none":
            args += ["--loopback", self.loopback.currentText()]
        if self.extra.text().strip():
            args += self.extra.text().split()
        return args

    def remote_online(self):
        if self.last_modem_status is None:
            return None
        age = time.monotonic() - self.last_modem_status[0]
        payload = self.last_modem_status[1]
        if payload.get("state") == "online" and age <= STALE_AFTER_S:
            return payload
        return None

    def start(self):
        log = self.bridge.log_message.emit
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        online = self.remote_online()
        if online is not None:
            log(f"Modem: not starting - a modem is already online on this topic (pid {online.get('pid')}); the radio can only be opened once")
            return
        exe = self.exe_edit.text().strip()
        if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            log(f"Modem: executable '{exe}' not found or not executable (build it with: cargo build --release)")
            return
        args = self.build_args()
        if args is None:
            return
        self.save_settings()
        self.stopping = False
        self._partial = ""
        log("Modem: starting: " + " ".join([exe] + args))
        self.process.setProgram(exe)
        self.process.setArguments(args)
        self.process.start()
        self.refresh_state()

    def stop(self):
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.stopping = True
        self._stop_token += 1
        token, pid = self._stop_token, self.process.processId()
        log = self.bridge.log_message.emit
        if self.bridge.is_connected:
            log("Modem: asking it to shut down cleanly (MQTT)")
            self.bridge.publish_modem_control("Shutdown")
            QTimer.singleShot(self.STOP_GRACE_MS, lambda: self._terminate_if_running(token, pid))
        else:
            log("Modem: not connected to the broker, sending SIGTERM")
            self._terminate_if_running(token, pid)
        self.refresh_state()

    def _still_the_same_stop(self, token, pid):
        return token == self._stop_token and self.process.processId() == pid and self.process.state() != QProcess.ProcessState.NotRunning

    def _terminate_if_running(self, token, pid):
        if self._still_the_same_stop(token, pid):
            self.bridge.log_message.emit("Modem: did not stop in time, terminating")
            self.process.terminate()
            QTimer.singleShot(self.TERMINATE_GRACE_MS, lambda: self._kill_if_running(token, pid))

    def _kill_if_running(self, token, pid):
        if self._still_the_same_stop(token, pid):
            self.bridge.log_message.emit("Modem: still running, killing")
            self.process.kill()

    def shutdown_for_exit(self):
        """Called when the GUI closes: stop our child if asked to, and wait."""
        self.save_settings()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        if not self.close_check.isChecked():
            return  # leave it running; the QProcess child is released below
        if self.bridge.is_connected:
            self.bridge.publish_modem_control("Shutdown")
            if self.process.waitForFinished(self.STOP_GRACE_MS):
                return
        self.process.terminate()
        if not self.process.waitForFinished(self.TERMINATE_GRACE_MS):
            self.process.kill()
            self.process.waitForFinished(2000)

    # ---- process events ----
    def on_started(self):
        self.bridge.log_message.emit(f"Modem: started (pid {self.process.processId()})")
        self.refresh_state()

    def on_finished(self, exit_code, exit_status):
        how = "stopped" if self.stopping else "EXITED UNEXPECTEDLY"
        crashed = " (crashed)" if exit_status == QProcess.ExitStatus.CrashExit else ""
        self.bridge.log_message.emit(f"Modem: {how}, exit code {exit_code}{crashed}")
        self.last_exit = (exit_code, self.stopping, exit_status == QProcess.ExitStatus.CrashExit)
        self._stop_token += 1  # cancel any pending terminate/kill fallback
        self.refresh_state()

    def on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.bridge.log_message.emit(f"Modem: failed to start: {self.process.errorString()}")
            self.last_exit = (-1, False, False)
        self.refresh_state()

    def on_output(self):
        text = self._partial + bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        *lines, self._partial = text.split("\n")
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if STATUS_LINE.match(line) and not self.verbose_check.isChecked():
                continue
            self.bridge.log_message.emit(f"[modem] {line}")

    # ---- state display ----
    def on_modem_status(self, payload: dict):
        self.last_modem_status = (time.monotonic(), payload)
        self.refresh_state()

    def refresh_state(self):
        st = self.process.state()
        running = st != QProcess.ProcessState.NotRunning
        if st == QProcess.ProcessState.Starting:
            self.state_label.set_value("starting ...", "warn")
        elif running:
            text = f"running (pid {self.process.processId()})"
            self.state_label.set_value("stopping ..." if self.stopping else text, "warn" if self.stopping else True)
        else:
            last = getattr(self, "last_exit", None)
            if last is None:
                self.state_label.set_value("stopped")
            else:
                code, clean, crashed = last
                if clean and code == 0:
                    self.state_label.set_value("stopped (clean exit)")
                else:
                    self.state_label.set_value(f"EXITED, code {code}{' (crashed)' if crashed else ''}" if not clean else f"stopped (exit code {code})", False)

        online = self.remote_online()
        if self.last_modem_status is None:
            self.remote_label.set_value("no modem status on MQTT")
        elif online is not None:
            mine = running and online.get("pid") == self.process.processId()
            self.remote_label.set_value(
                f"online, pid {online.get('pid')}{' (started by this GUI)' if mine else ' (started elsewhere)'}, up {online.get('uptime_s', 0):.0f} s, "
                f"TX {'on' if online.get('tx_enabled') else 'off'} RX {'on' if online.get('rx_enabled') else 'off'}, loopback {online.get('loopback')}, KISS {online.get('kiss_port')}",
                True,
            )
        elif self.last_modem_status[1].get("state") == "offline":
            self.remote_label.set_value("offline", None)
        else:
            self.remote_label.set_value("status stale", False)

        self.start_btn.setEnabled(not running and online is None)
        self.stop_btn.setEnabled(running)


class LinkPanel(QGroupBox):
    """Modulation and symbol-rate control for the TX, the RX or both. The modem
    rebuilds the affected chain, so the link drops and the receiver
    re-acquires; both ends must be set to the same values to link."""

    def __init__(self, bridge: MqttBridge):
        super().__init__("Link settings (modulation / symbol rate)")
        self.bridge = bridge
        self.tx_status = {}
        self.rx_status = {}

        self.modulation = QComboBox()
        self.modulation.addItems(list(MODULATIONS))
        self.modulation.setCurrentText("8PSK")
        self.rate_edit = QLineEdit("16000")
        self.rate_edit.setFixedWidth(80)
        self.rate_edit.returnPressed.connect(lambda: self.apply(True, True))
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: gray;")
        self.rate_edit.textChanged.connect(self.update_info)
        self.modulation.currentTextChanged.connect(self.update_info)

        both_btn = QPushButton("Apply to TX + RX")
        tx_btn = QPushButton("TX only")
        rx_btn = QPushButton("RX only")
        both_btn.clicked.connect(lambda: self.apply(True, True))
        tx_btn.clicked.connect(lambda: self.apply(True, False))
        rx_btn.clicked.connect(lambda: self.apply(False, True))

        row = QHBoxLayout()
        row.addWidget(QLabel("Modulation:"))
        row.addWidget(self.modulation)
        row.addWidget(QLabel("Symbol rate (sym/s):"))
        row.addWidget(self.rate_edit)
        row.addStretch()
        buttons = QHBoxLayout()
        for b in (both_btn, tx_btn, rx_btn):
            buttons.addWidget(b)
        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addWidget(self.info_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.bridge.tx_status.connect(lambda s: setattr(self, "tx_status", s))
        self.bridge.rx_status.connect(lambda s: setattr(self, "rx_status", s))
        self.update_info()

    def requested_rate(self):
        try:
            rate = float(self.rate_edit.text().strip())
        except ValueError:
            return None
        return rate if MIN_SYMBOL_RATE <= rate <= MAX_SYMBOL_RATE else None

    def update_info(self, *_):
        rate = self.requested_rate()
        if rate is None:
            self.info_label.setText(f"symbol rate must be {MIN_SYMBOL_RATE:.0f}..{MAX_SYMBOL_RATE:.0f}")
            return
        wire, points = MODULATIONS[self.modulation.currentText()]
        bits = points.bit_length() - 1
        occupied = rate * (1 + ROLLOFF)
        sdr = "800 kSPS" if rate <= LOW_PROFILE_MAX_RATE else "4 MSPS"
        self.info_label.setText(
            f"{rate * bits / 1e3:.1f} kbps on air / {rate * bits / 2e3:.1f} kbps data, "
            f"occupied {occupied / 1e3:.1f} kHz, SDR {sdr} (keep the freq shift above {occupied / 2e3:.1f} kHz)"
        )

    def apply(self, to_tx, to_rx):
        log = self.bridge.log_message.emit
        rate = self.requested_rate()
        if rate is None:
            log(f"Symbol rate must be a number from {MIN_SYMBOL_RATE:.0f} to {MAX_SYMBOL_RATE:.0f}")
            return
        wire = MODULATIONS[self.modulation.currentText()][0]
        for enabled, status, publish, name in (
            (to_tx, self.tx_status, self.bridge.publish_tx_control, "TX"),
            (to_rx, self.rx_status, self.bridge.publish_rx_control, "RX"),
        ):
            if not enabled:
                continue
            if status.get("modulation") != wire:
                publish({"SetModulation": wire})
            # A rate change rebuilds the chain (dropping the link): skip a no-op.
            if status.get("symbol_rate_hz") != rate:
                publish({"SetSymbolRateHz": rate})
            shift = status.get("freq_shift_hz")
            if shift is not None and abs(shift) < rate * (1 + ROLLOFF) / 2:
                log(f"Warning: {name} freq shift {shift:+.0f} Hz is inside the +-{rate * (1 + ROLLOFF) / 2:.0f} Hz occupied band - raise it (TX and RX must match)")


class TxPanel(QGroupBox):
    def __init__(self, bridge: MqttBridge):
        super().__init__("Transmitter")
        self.bridge = bridge
        self.last_status = None

        self.live_label = StatusLabel("no status yet")
        self.source_label = StatusLabel("-")
        self.modulation_label = StatusLabel("-")
        self.frequency_label = StatusLabel("-")
        self.shift_label = StatusLabel("-")
        self.gain_label = StatusLabel("-")
        self.rates_label = StatusLabel("-")
        self.pattern_label = StatusLabel("-")
        self.bits_sent_label = StatusLabel("-")
        self.errors_sent_label = StatusLabel("-")
        self.kiss_client_label = StatusLabel("-")
        self.frames_sent_label = StatusLabel("-")
        self.uptime_label = StatusLabel("-")

        form = QFormLayout()
        form.addRow("Link:", self.live_label)
        form.addRow("Source:", self.source_label)
        form.addRow("Modulation:", self.modulation_label)
        form.addRow("Frequency:", self.frequency_label)
        form.addRow("Freq shift (DSP):", self.shift_label)
        form.addRow("Gain (dB):", self.gain_label)
        form.addRow("Rate on air / data:", self.rates_label)
        form.addRow("BERT pattern:", self.pattern_label)
        form.addRow("BERT bits sent:", self.bits_sent_label)
        form.addRow("BERT errors injected:", self.errors_sent_label)
        form.addRow("KISS client connected:", self.kiss_client_label)
        form.addRow("AX.25 frames sent:", self.frames_sent_label)
        form.addRow("Uptime (s):", self.uptime_label)

        bert_btn = QPushButton("Source: BERT")
        data_btn = QPushButton("Source: DATA")
        inject_btn = QPushButton("Inject bit error")
        reset_btn = QPushButton("Reset TX BERT")
        bert_btn.clicked.connect(lambda: self.bridge.publish_tx_control({"SetSource": "Bert"}))
        data_btn.clicked.connect(lambda: self.bridge.publish_tx_control({"SetSource": "Data"}))
        inject_btn.clicked.connect(lambda: self.bridge.publish_tx_control({"Bert": "InjectError"}))
        reset_btn.clicked.connect(lambda: self.bridge.publish_tx_control({"Bert": "Reset"}))
        buttons = QGridLayout()
        for k, b in enumerate((bert_btn, data_btn, inject_btn, reset_btn)):
            buttons.addWidget(b, k // 2, k % 2)

        self.gain_row = SettingRow("gain dB", "Set gain", self.on_set_gain)
        self.freq_row = SettingRow("frequency Hz", "Set frequency", self.on_set_frequency, width=130)
        self.shift_row = SettingRow("shift Hz", "Set freq shift", self.on_set_shift)
        settings = QFormLayout()
        settings.addRow("Gain:", self.gain_row)
        settings.addRow("Frequency:", self.freq_row)
        settings.addRow("Freq shift (match RX):", self.shift_row)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addLayout(settings)
        layout.addStretch()
        self.setLayout(layout)

        self.bridge.tx_status.connect(self.on_status)

    def on_set_gain(self):
        v = parse_number(self.gain_row.text(), "gain", self.bridge.log_message.emit)
        if v is not None:
            self.bridge.publish_tx_control({"SetGainDb": int(round(v))})

    def on_set_frequency(self):
        v = parse_number(self.freq_row.text(), "frequency", self.bridge.log_message.emit)
        if v is not None and v > 0:
            self.bridge.publish_tx_control({"SetFrequencyHz": int(round(v))})

    def on_set_shift(self):
        v = parse_number(self.shift_row.text(), "frequency shift", self.bridge.log_message.emit)
        if v is not None:
            self.bridge.publish_tx_control({"SetFreqShiftHz": v})

    def on_status(self, status: dict):
        self.last_status = time.monotonic()
        src = status.get("source", "-")
        self.source_label.set_value(str(src), True if src == "Data" else None)
        self.modulation_label.set_value(LABELS.get(status.get("modulation"), "-"))
        freq = status.get("frequency_hz")
        self.frequency_label.set_value("-" if freq is None else f"{freq / 1e6:.6f} MHz")
        shift = status.get("freq_shift_hz")
        self.shift_label.set_value("-" if shift is None else f"{shift:+.1f} Hz")
        self.gain_label.set_value(str(status.get("gain_db", "-")))
        air, data, sym = status.get("bit_rate_bps"), status.get("info_bit_rate_bps"), status.get("symbol_rate_hz")
        sdr = status.get("sample_rate_hz")
        self.rates_label.set_value("-" if air is None else f"{air / 1e3:.1f} / {data / 1e3:.1f} kbps  ({sym / 1e3:.1f} ksym/s, SDR {'-' if sdr is None else f'{sdr / 1e6:g} MSPS'})")
        self.pattern_label.set_value(str(status.get("pattern", "-")))
        bert = status.get("bert", {})
        self.bits_sent_label.set_value(str(bert.get("bits_sent", "-")))
        self.errors_sent_label.set_value(str(bert.get("bit_errors_sent", "-")))
        self.kiss_client_label.set_value(str(status.get("kiss_client_connected", "-")))
        self.frames_sent_label.set_value(str(status.get("frames_sent", "-")))
        self.uptime_label.set_value(f"{status.get('uptime_s', 0.0):.1f}")

    def refresh_liveness(self):
        age = None if self.last_status is None else time.monotonic() - self.last_status
        if age is None:
            self.live_label.set_value("no status yet", "warn")
        elif age > STALE_AFTER_S:
            self.live_label.set_value(f"STALE ({age:.0f} s)", False)
        else:
            self.live_label.set_value("live", True)


class RxPanel(QGroupBox):
    def __init__(self, bridge: MqttBridge, trends):
        super().__init__("Receiver")
        self.bridge = bridge
        self.trends = trends
        self.last_status = None
        self.t0 = time.monotonic()
        self._updating_controls = False

        self.live_label = StatusLabel("no status yet")
        self.modulation_label = StatusLabel("-")
        self.modulation = "Psk8"

        # --- Lock ---
        self.carrier_locked_label = StatusLabel("-")
        self.bert_state_label = StatusLabel("-")
        self.bert_rotation_label = StatusLabel("-")
        self.frame_rotation_label = StatusLabel("-")
        self.carrier_err_label = StatusLabel("-")
        self.timing_label = StatusLabel("-")
        lock = QFormLayout()
        lock.addRow("Modulation:", self.modulation_label)
        lock.addRow("Carrier locked (verified):", self.carrier_locked_label)
        lock.addRow("BERT state:", self.bert_state_label)
        lock.addRow("BERT rotation:", self.bert_rotation_label)
        lock.addRow("HDLC rotation:", self.frame_rotation_label)
        lock.addRow("Carrier |phase err| (rad):", self.carrier_err_label)
        lock.addRow("Timing (samples/symbol):", self.timing_label)
        lock_box = QGroupBox("Lock")
        lock_box.setLayout(lock)

        # --- Signal ---
        self.level_label = StatusLabel("-")
        self.snr_label = StatusLabel("-")
        self.err_std_label = StatusLabel("-")
        self.total_offset_label = StatusLabel("-")
        self.lo_retune_label = StatusLabel("-")
        self.carrier_freq_label = StatusLabel("-")
        signal = QFormLayout()
        signal.addRow("Input level (dBFS):", self.level_label)
        signal.addRow("SNR (Es/N0, dB):", self.snr_label)
        signal.addRow("Symbol error std:", self.err_std_label)
        signal.addRow("Total freq offset (Hz):", self.total_offset_label)
        signal.addRow("  = LO retune (Hz):", self.lo_retune_label)
        signal.addRow("  + carrier residual (Hz):", self.carrier_freq_label)
        signal_box = QGroupBox("Signal")
        signal_box.setLayout(signal)

        # --- Errors ---
        self.pre_fec_label = StatusLabel("-")
        self.post_fec_label = StatusLabel("-")
        self.bits_received_label = StatusLabel("-")
        self.errors_received_label = StatusLabel("-")
        self.sync_loss_label = StatusLabel("-")
        self.frames_label = StatusLabel("-")
        self.bad_frames_label = StatusLabel("-")
        errors = QFormLayout()
        errors.addRow("Pre-FEC BER (BERT / frames):", self.pre_fec_label)
        errors.addRow("Post-FEC BER (BERT):", self.post_fec_label)
        errors.addRow("BERT bits decoded:", self.bits_received_label)
        errors.addRow("BERT bit errors:", self.errors_received_label)
        errors.addRow("BERT sync loss (latched):", self.sync_loss_label)
        errors.addRow("Frames received:", self.frames_label)
        errors.addRow("Bad frames (FCS, locked branch):", self.bad_frames_label)
        errors_box = QGroupBox("Errors")
        errors_box.setLayout(errors)

        # --- Link ---
        self.frequency_label = StatusLabel("-")
        self.shift_label = StatusLabel("-")
        self.gain_label = StatusLabel("-")
        self.loops_label = StatusLabel("-")
        self.pattern_label = StatusLabel("-")
        self.kiss_client_label = StatusLabel("-")
        self.uptime_label = StatusLabel("-")
        link = QFormLayout()
        link.addRow("Frequency:", self.frequency_label)
        link.addRow("Freq shift (DSP):", self.shift_label)
        link.addRow("Gain (dB):", self.gain_label)
        link.addRow("Carrier / timing loop BW:", self.loops_label)
        link.addRow("BERT pattern:", self.pattern_label)
        link.addRow("KISS client connected:", self.kiss_client_label)
        link.addRow("Uptime (s):", self.uptime_label)
        link_box = QGroupBox("Link")
        link_box.setLayout(link)

        sections = QGridLayout()
        sections.addWidget(lock_box, 0, 0)
        sections.addWidget(signal_box, 0, 1)
        sections.addWidget(errors_box, 1, 0)
        sections.addWidget(link_box, 1, 1)

        # --- Commands ---
        reset_btn = QPushButton("Reset RX BERT")
        reset_stats_btn = QPushButton("Reset BERT stats")
        reacquire_btn = QPushButton("Reacquire")
        reacquire_btn.setToolTip("Reset the receiver's timing and carrier loops and acquire again from scratch")
        reacquire_btn.clicked.connect(lambda: self.bridge.publish_rx_control("Reacquire"))
        reset_btn.clicked.connect(lambda: self.bridge.publish_rx_control({"Bert": "Reset"}))
        reset_stats_btn.clicked.connect(lambda: self.bridge.publish_rx_control({"Bert": "ResetStats"}))
        buttons = QHBoxLayout()
        buttons.addWidget(reset_btn)
        buttons.addWidget(reset_stats_btn)
        buttons.addWidget(reacquire_btn)
        self.search_box = QCheckBox("Frequency acquisition (spectral search)")
        self.search_box.toggled.connect(self.on_search_toggled)
        buttons.addWidget(self.search_box)
        buttons.addStretch()

        self.gain_row = SettingRow("gain dB", "Set gain", self.on_set_gain)
        self.freq_row = SettingRow("frequency Hz", "Set frequency", self.on_set_frequency, width=130)
        self.shift_row = SettingRow("shift Hz", "Set freq shift", self.on_set_shift)
        self.carrier_bw_row = SettingRow("carrier BW Hz", "Set", self.on_set_carrier_bw)
        self.timing_bw_row = SettingRow("timing BW", "Set", self.on_set_timing_bw)
        settings = QFormLayout()
        settings.addRow("Gain:", self.gain_row)
        settings.addRow("Frequency:", self.freq_row)
        settings.addRow("Freq shift (match TX):", self.shift_row)
        settings.addRow("Carrier loop BW (default 100):", self.carrier_bw_row)
        settings.addRow("Timing loop BW (default 0.02):", self.timing_bw_row)

        layout = QVBoxLayout()
        top = QFormLayout()
        top.addRow("Link:", self.live_label)
        layout.addLayout(top)
        layout.addLayout(sections)
        layout.addLayout(buttons)
        layout.addLayout(settings)
        self.setLayout(layout)

        self.bridge.rx_status.connect(self.on_status)

    # ---- commands ----
    def on_set_gain(self):
        v = parse_number(self.gain_row.text(), "gain", self.bridge.log_message.emit)
        if v is not None:
            self.bridge.publish_rx_control({"SetGainDb": int(round(v))})

    def on_set_frequency(self):
        v = parse_number(self.freq_row.text(), "frequency", self.bridge.log_message.emit)
        if v is not None and v > 0:
            self.bridge.publish_rx_control({"SetFrequencyHz": int(round(v))})

    def on_set_shift(self):
        v = parse_number(self.shift_row.text(), "frequency shift", self.bridge.log_message.emit)
        if v is not None:
            self.bridge.publish_rx_control({"SetFreqShiftHz": v})

    def on_set_carrier_bw(self):
        v = parse_number(self.carrier_bw_row.text(), "carrier bandwidth", self.bridge.log_message.emit)
        if v is not None and v > 0:
            self.bridge.publish_rx_control({"SetCarrierBandwidthHz": v})

    def on_set_timing_bw(self):
        v = parse_number(self.timing_bw_row.text(), "timing bandwidth", self.bridge.log_message.emit)
        if v is not None and v > 0:
            self.bridge.publish_rx_control({"SetTimingBandwidth": v})

    def on_search_toggled(self, checked):
        if not self._updating_controls:  # only user toggles, not status echoes
            self.bridge.publish_rx_control({"SetSearchEnabled": bool(checked)})

    # ---- telemetry ----
    def on_status(self, status: dict):
        now = time.monotonic()
        self.last_status = now
        t = now - self.t0

        locked = status.get("carrier_locked")
        self.carrier_locked_label.set_value("LOCKED" if locked else "searching", True if locked else ("warn" if locked is not None else None))
        bert = status.get("bert", {})
        state = bert.get("locked_state", "-")
        self.bert_state_label.set_value(str(state), True if state == "Synced" else None)
        self.modulation = status.get("modulation", self.modulation)
        self.modulation_label.set_value(LABELS.get(self.modulation, "-"))
        self.bert_rotation_label.set_value(rotation_text(status.get("bert_rotation"), self.modulation))
        self.frame_rotation_label.set_value(rotation_text(status.get("frame_rotation"), self.modulation))
        self.carrier_err_label.set_value(fmt_num(status.get("carrier_phase_error"), ".3f"))
        self.timing_label.set_value(fmt_num(status.get("timing_sps"), ".4f"))

        level = status.get("level_dbfs")
        self.level_label.set_value(fmt_num(level, ".1f"))
        snr, std = status.get("snr_db"), status.get("symbol_error_std")
        self.snr_label.set_value(fmt_num(snr, ".1f"), None if snr is None else (True if snr >= 14 else ("warn" if snr >= 11 else False)))
        self.err_std_label.set_value(fmt_num(std, ".4f"))
        total = status.get("total_offset_hz")
        self.total_offset_label.set_value(fmt_num(total, "+.1f"))
        self.lo_retune_label.set_value(fmt_num(status.get("lo_search_offset_hz"), "+.1f"))
        self.carrier_freq_label.set_value(fmt_num(status.get("carrier_frequency_hz"), "+.1f"))

        pre_b, pre_f = status.get("bert_pre_fec_ber"), status.get("frame_pre_fec_ber")
        self.pre_fec_label.set_value(f"{fmt_ber(pre_b)} / {fmt_ber(pre_f)}")
        bits = bert.get("bits_received", 0)
        errors = bert.get("bit_errors_received", 0)
        have_post = bool(bits) and state == "Synced"
        self.post_fec_label.set_value(f"{errors / bits:.2e}" if have_post else "-", (errors == 0) if have_post else None)
        self.bits_received_label.set_value(str(bits))
        self.errors_received_label.set_value(str(errors))
        loss = bert.get("sync_loss")
        self.sync_loss_label.set_value("-" if loss is None else str(loss), None if loss is None else (not loss))
        self.frames_label.set_value(str(status.get("frames_received", "-")))
        self.bad_frames_label.set_value(str(status.get("bad_frames", "-")))

        freq = status.get("frequency_hz")
        self.frequency_label.set_value("-" if freq is None else f"{freq / 1e6:.6f} MHz")
        shift = status.get("freq_shift_hz")
        self.shift_label.set_value("-" if shift is None else f"{shift:+.1f} Hz")
        self.gain_label.set_value(str(status.get("gain_db", "-")))
        self.loops_label.set_value(f"{fmt_num(status.get('carrier_bandwidth_hz'), '.0f')} Hz / {fmt_num(status.get('timing_bandwidth'), '.3f')}")
        self.pattern_label.set_value(str(status.get("pattern", "-")))
        self.kiss_client_label.set_value(str(status.get("kiss_client_connected", "-")))
        self.uptime_label.set_value(f"{status.get('uptime_s', 0.0):.1f}")

        enabled = status.get("search_enabled")
        if enabled is not None and enabled != self.search_box.isChecked():
            self._updating_controls = True
            self.search_box.setChecked(bool(enabled))
            self._updating_controls = False

        # Trends: SNR and pre-FEC BER only exist while locked (gaps otherwise).
        self.trends["snr"].add(t, snr)
        self.trends["offset"].add(t, total if locked else None)
        best_pre = pre_b if pre_b is not None else pre_f
        self.trends["ber"].add(t, None if best_pre is None else math.log10(max(best_pre, BER_FLOOR)))
        self.trends["level"].add(t, level)

    def refresh_liveness(self):
        age = None if self.last_status is None else time.monotonic() - self.last_status
        if age is None:
            self.live_label.set_value("no status yet", "warn")
        elif age > STALE_AFTER_S:
            self.live_label.set_value(f"STALE ({age:.0f} s)", False)
        else:
            self.live_label.set_value("live", True)


class MainWindow(QMainWindow):
    def __init__(self, broker_host: str, broker_port: int, base_topic: str):
        super().__init__()
        self.setWindowTitle("PSK modem control")

        self.bridge = MqttBridge(base_topic)
        self.bridge.connection_changed.connect(self.on_connection_changed)
        self.bridge.log_message.connect(self.on_log_message)
        self.bridge.rx_status.connect(self.on_rx_summary)

        self.host_edit = QLineEdit(broker_host)
        self.port_edit = QLineEdit(str(broker_port))
        self.port_edit.setFixedWidth(60)
        self.topic_edit = QLineEdit(base_topic)
        self.topic_edit.setFixedWidth(100)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.status_label = QLabel("disconnected")

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Broker:"))
        conn_row.addWidget(self.host_edit)
        conn_row.addWidget(QLabel(":"))
        conn_row.addWidget(self.port_edit)
        conn_row.addWidget(QLabel("Base topic:"))
        conn_row.addWidget(self.topic_edit)
        conn_row.addWidget(self.connect_btn)
        conn_row.addWidget(self.status_label)
        conn_row.addStretch()

        self.banner = QLabel("waiting for receiver status ...")
        self.banner.setStyleSheet("color: gray;")
        self.summary_label = QLabel("no receiver status yet")
        font = QFont()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.summary_label.setFont(font)

        self.trends = {
            "snr": TrendChart("SNR (Es/N0)", "dB"),
            "offset": TrendChart("Total frequency offset", "Hz"),
            "ber": TrendChart("Pre-FEC BER (log10)", "log10", y_min=math.log10(BER_FLOOR), y_max=0.0),
            "level": TrendChart("RX input level", "dBFS"),
        }
        self.constellation = ConstellationPlot()
        self.bridge.rx_symbols.connect(self.constellation.update_symbols)
        self.bridge.rx_status.connect(lambda s: self.constellation.set_modulation(s.get("modulation", "Psk8")))
        self.modem_panel = ModemPanel(self.bridge, self.current_connection)
        self.link_panel = LinkPanel(self.bridge)
        self.tx_panel = TxPanel(self.bridge)
        self.rx_panel = RxPanel(self.bridge, self.trends)

        # Three columns so the window fits a 1920x1080 screen: modem + TX on
        # the left, RX in the middle, constellation + trends on the right.
        left = QVBoxLayout()
        left.addWidget(self.modem_panel)
        left.addWidget(self.link_panel)
        left.addWidget(self.tx_panel)
        left.addStretch()

        plots = QGridLayout()
        plots.addWidget(self.constellation, 0, 0, 1, 2)
        plots.addWidget(self.trends["snr"], 1, 0)
        plots.addWidget(self.trends["offset"], 1, 1)
        plots.addWidget(self.trends["ber"], 2, 0)
        plots.addWidget(self.trends["level"], 2, 1)
        plots.setRowStretch(0, 2)
        plots.setRowStretch(1, 1)
        plots.setRowStretch(2, 1)

        columns = QHBoxLayout()
        columns.addLayout(left, 0)
        columns.addWidget(self.rx_panel, 0)
        columns.addLayout(plots, 1)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(80)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(conn_row)
        layout.addWidget(self.banner)
        layout.addWidget(self.summary_label)
        layout.addLayout(columns, 1)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_view)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.liveness_timer = QTimer(self)
        self.liveness_timer.timeout.connect(self.refresh_liveness)
        self.liveness_timer.start(1000)

        self.bridge.connect_to(broker_host, broker_port, base_topic)

    def on_rx_summary(self, s: dict):
        sym, air, data = s.get("symbol_rate_hz"), s.get("bit_rate_bps"), s.get("info_bit_rate_bps")
        if sym is not None and air is not None and data is not None:
            self.banner.setText(
                f"{LABELS.get(s.get('modulation'), '?')} (Gray)  |  {sym / 1e3:g} ksym/s  |  RRC {ROLLOFF}  |  "
                f"{sym * (1 + ROLLOFF) / 1e3:.1f} kHz occupied  |  K=7 r=1/2 FEC  |  {air / 1e3:g} kbps on air, {data / 1e3:g} kbps data"
            )
        locked = s.get("carrier_locked")
        snr = s.get("snr_db")
        parts = ["RX LOCKED" if locked else "RX searching"]
        if locked:
            parts.append(f"SNR {snr:.1f} dB" if snr is not None else "SNR -")
            parts.append(f"offset {s.get('total_offset_hz', 0.0):+.0f} Hz")
            pre = s.get("bert_pre_fec_ber") if s.get("bert_pre_fec_ber") is not None else s.get("frame_pre_fec_ber")
            parts.append(f"pre-FEC BER {fmt_ber(pre)}")
        parts.append(f"frames {s.get('frames_received', 0)}")
        self.summary_label.setText("   |   ".join(parts))
        self.summary_label.setStyleSheet(OK_STYLE if locked else WARN_STYLE)

    def current_connection(self):
        """(host, port, base topic) as currently typed in the connection bar -
        the modem is started pointing at the same broker the GUI uses."""
        try:
            port = int(self.port_edit.text().strip())
        except ValueError:
            port = 1883
        return self.host_edit.text().strip() or "127.0.0.1", port, self.topic_edit.text().strip() or "psk8"

    def refresh_liveness(self):
        self.modem_panel.refresh_state()
        self.tx_panel.refresh_liveness()
        self.rx_panel.refresh_liveness()
        if self.rx_panel.last_status is None or time.monotonic() - self.rx_panel.last_status > STALE_AFTER_S:
            self.summary_label.setText("receiver status stale")
            self.summary_label.setStyleSheet(BAD_STYLE)

    def on_connect_clicked(self):
        self.bridge.disconnect()
        host = self.host_edit.text().strip()
        port = int(self.port_edit.text().strip())
        base_topic = self.topic_edit.text().strip() or "psk8"
        self.bridge.connect_to(host, port, base_topic)

    def on_connection_changed(self, connected: bool, message: str):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {'green' if connected else 'red'}")

    def on_log_message(self, text: str):
        self.log_view.append(text)

    def closeEvent(self, event):
        self.modem_panel.shutdown_for_exit()
        self.bridge.disconnect()
        super().closeEvent(event)


def main():
    broker_host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    broker_port = int(sys.argv[2]) if len(sys.argv) > 2 else 1883
    base_topic = sys.argv[3] if len(sys.argv) > 3 else "psk8"

    app = QApplication(sys.argv)
    window = MainWindow(broker_host, broker_port, base_topic)
    window.resize(1900, 960)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
