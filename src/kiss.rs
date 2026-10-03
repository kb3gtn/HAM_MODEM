//! KISS: the simple byte-stuffed host<->TNC protocol that carries opaque
//! AX.25 frame content (see `hdlc.rs` - this project never interprets that
//! content itself). Exposed over TCP on port 8001 by convention, matching
//! the de facto standard used by direwolf/soundmodem, rather than a serial
//! device - a natural fit for a headless SDR host.
//!
//! Only the data command (0x00) is acted on; other KISS commands (TXDELAY,
//! persistence, slot time, full duplex, set hardware) are accepted and
//! silently ignored - this project's "radio" is a continuously-transmitting
//! SDR with no analog PTT relay to time, so those settings don't apply here.
//!
//! V1 supports one connected client at a time (a new connection replaces
//! any existing one) - direwolf supports multiple simultaneous KISS clients
//! broadcasting/merging, which would be a reasonable future enhancement but
//! isn't needed for a first working interface.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};

const FEND: u8 = 0xC0;
const FESC: u8 = 0xDB;
const TFEND: u8 = 0xDC;
const TFESC: u8 = 0xDD;
const DATA_COMMAND: u8 = 0x00;

/// Encodes one AX.25 frame's opaque bytes as a complete KISS frame (data
/// command, port 0) ready to write to the host, including the delimiting
/// `FEND` bytes.
pub fn encode(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 4);
    out.push(FEND);
    out.push(DATA_COMMAND);
    for &b in payload {
        match b {
            FEND => {
                out.push(FESC);
                out.push(TFEND);
            }
            FESC => {
                out.push(FESC);
                out.push(TFESC);
            }
            other => out.push(other),
        }
    }
    out.push(FEND);
    out
}

/// Streaming KISS byte decoder - feed it raw bytes as they arrive from the
/// host (over TCP, in whatever chunk sizes the socket happens to deliver),
/// get back complete data-frame payloads as they're found.
pub struct KissDecoder {
    buf: Vec<u8>,
    escaped: bool,
}

impl KissDecoder {
    pub fn new() -> Self {
        KissDecoder {
            buf: Vec::new(),
            escaped: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        for &b in bytes {
            if self.escaped {
                self.escaped = false;
                match b {
                    TFEND => self.buf.push(FEND),
                    TFESC => self.buf.push(FESC),
                    other => self.buf.push(other), // malformed escape - lenient passthrough
                }
                continue;
            }
            match b {
                FEND => {
                    // Consecutive FENDs (empty buf) are a harmless, common
                    // KISS idiom - just skip them rather than emitting
                    // empty frames.
                    if !self.buf.is_empty() {
                        if self.buf[0] & 0x0F == DATA_COMMAND {
                            frames.push(self.buf[1..].to_vec());
                        }
                        self.buf.clear();
                    }
                }
                FESC => self.escaped = true,
                other => self.buf.push(other),
            }
        }
        frames
    }
}

impl Default for KissDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A TCP KISS server: accepts a client connection, decodes frames it sends
/// (available via `try_recv_frame`, non-blocking - matching the project's
/// established TX pacing rule that the bit source must never block), and
/// encodes/writes frames handed to `send_frame` out to whichever client is
/// currently connected (a no-op if none is).
pub struct KissServer {
    outgoing_client: Arc<Mutex<Option<TcpStream>>>,
    incoming_frames: Receiver<Vec<u8>>,
}

impl KissServer {
    pub fn start(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let outgoing_client: Arc<Mutex<Option<TcpStream>>> = Arc::new(Mutex::new(None));
        let (frame_tx, frame_rx) = bounded(64);

        let accept_client = outgoing_client.clone();
        thread::spawn(move || {
            Self::accept_loop(listener, accept_client, frame_tx);
        });

        Ok(KissServer {
            outgoing_client,
            incoming_frames: frame_rx,
        })
    }

    fn accept_loop(
        listener: TcpListener,
        current_client: Arc<Mutex<Option<TcpStream>>>,
        frame_tx: Sender<Vec<u8>>,
    ) {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let Ok(read_half) = stream.try_clone() else {
                continue;
            };
            *current_client.lock().expect("kiss client mutex poisoned") = Some(stream);

            let frame_tx = frame_tx.clone();
            thread::spawn(move || Self::client_read_loop(read_half, frame_tx));
        }
    }

    fn client_read_loop(mut stream: TcpStream, frame_tx: Sender<Vec<u8>>) {
        let mut decoder = KissDecoder::new();
        let mut buf = [0u8; 4096];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => return, // client disconnected
                Ok(n) => {
                    for frame in decoder.feed(&buf[..n]) {
                        let _ = frame_tx.try_send(frame); // drop under backpressure, never block
                    }
                }
            }
        }
    }

    /// Non-blocking: the next frame received from the host, if any is
    /// queued.
    pub fn try_recv_frame(&self) -> Option<Vec<u8>> {
        self.incoming_frames.try_recv().ok()
    }

    /// Whether a host is currently connected - useful telemetry (a modem
    /// that's "up" but has nothing listening/feeding it looks very
    /// different operationally from one that's actively bridged).
    pub fn has_client(&self) -> bool {
        self.outgoing_client
            .lock()
            .expect("kiss client mutex poisoned")
            .is_some()
    }

    /// Sends a frame received off the air to the currently connected
    /// client, if any - silently does nothing otherwise.
    pub fn send_frame(&self, payload: &[u8]) {
        let mut guard = self
            .outgoing_client
            .lock()
            .expect("kiss client mutex poisoned");
        if let Some(stream) = guard.as_mut() {
            if stream.write_all(&encode(payload)).is_err() {
                *guard = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_then_decode_recovers_the_payload() {
        let payload = b"hello KISS".to_vec();
        let encoded = encode(&payload);

        let mut decoder = KissDecoder::new();
        let frames = decoder.feed(&encoded);

        assert_eq!(frames, vec![payload]);
    }

    #[test]
    fn escapes_fend_and_fesc_bytes_in_the_payload() {
        let payload = vec![0x01, FEND, 0x02, FESC, 0x03];
        let encoded = encode(&payload);

        // The encoded form must not contain a bare FEND/FESC anywhere
        // except the two framing FENDs at the very start and end.
        let inner = &encoded[2..encoded.len() - 1]; // skip [FEND, cmd] and the trailing FEND
        assert!(
            !inner.contains(&FEND),
            "raw FEND leaked into the encoded payload"
        );

        let mut decoder = KissDecoder::new();
        let frames = decoder.feed(&encoded);
        assert_eq!(frames, vec![payload]);
    }

    #[test]
    fn decodes_multiple_frames_fed_across_separate_calls() {
        let a = b"first".to_vec();
        let b = b"second".to_vec();
        let mut decoder = KissDecoder::new();

        let encoded_a = encode(&a);
        let (part1, part2) = encoded_a.split_at(encoded_a.len() / 2);
        let mut frames = decoder.feed(part1);
        assert!(
            frames.is_empty(),
            "shouldn't produce a frame mid-way through encoding"
        );
        frames.extend(decoder.feed(part2));
        frames.extend(decoder.feed(&encode(&b)));

        assert_eq!(frames, vec![a, b]);
    }

    #[test]
    fn ignores_non_data_commands() {
        // command nibble 0x01 = TXDELAY, not a data frame - should be
        // silently accepted and ignored, not passed through as a payload.
        let mut raw = vec![FEND, 0x01, 0x0A];
        raw.push(FEND);
        let mut decoder = KissDecoder::new();
        let frames = decoder.feed(&raw);
        assert!(frames.is_empty());
    }

    #[test]
    fn consecutive_fends_do_not_produce_empty_frames() {
        let mut decoder = KissDecoder::new();
        let frames = decoder.feed(&[FEND, FEND, FEND]);
        assert!(frames.is_empty());
    }

    #[test]
    fn tcp_server_round_trips_frames_with_a_real_client() {
        use std::io::Read as _;
        use std::net::TcpStream;
        use std::time::Duration;

        // KissServer::start doesn't expose the port it bound, so grab a free
        // one via a throwaway listener first, then point the real server at
        // it by address (a small, standard test-isolation trick - accepts a
        // theoretical race against another process grabbing the same port
        // in between, negligible in practice for a test).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener); // free the port again for KissServer::start to bind

        let server = KissServer::start(&addr.to_string()).expect("bind kiss server");
        thread::sleep(Duration::from_millis(50)); // let the accept thread start listening

        let mut client = TcpStream::connect(addr).expect("connect to kiss server");
        thread::sleep(Duration::from_millis(50)); // let the server register the connection

        // Host -> radio direction.
        let host_payload = b"from the host".to_vec();
        client.write_all(&encode(&host_payload)).unwrap();
        thread::sleep(Duration::from_millis(100));
        let received = server.try_recv_frame();
        assert_eq!(received, Some(host_payload));

        // Radio -> host direction.
        let radio_payload = b"from the radio".to_vec();
        server.send_frame(&radio_payload);
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut read_buf = vec![0u8; 256];
        let n = client.read(&mut read_buf).expect("read from kiss server");
        let mut decoder = KissDecoder::new();
        let frames = decoder.feed(&read_buf[..n]);
        assert_eq!(frames, vec![radio_payload]);
    }
}
