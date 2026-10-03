//! Standalone runnable BERT test node: runs a TX BERT and RX BERT connected
//! in a software loopback (no bladeRF yet), and bridges their control/status
//! crossbeam channels to MQTT so an external tool (mosquitto_pub/sub, or
//! anything else) can drive commands and watch telemetry.
//!
//! Usage: bert_node [broker_host] [broker_port] [initial_bit_rate_bps]
//! Defaults: 127.0.0.1 1883 1000
//!
//! Topics:
//!   bpsk/tx/control  (subscribe, QoS1) - send a JSON TxControl
//!   bpsk/tx/status   (publish, QoS0, retained) - JSON TxStatus
//!   bpsk/rx/control  (subscribe, QoS1) - send a JSON RxControl
//!   bpsk/rx/status   (publish, QoS0, retained) - JSON RxStatus

use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, select, unbounded};
use rumqttc::{Client, Event, MqttOptions, Packet, QoS};

use ham_modem::prbs::PrbsPattern;
use ham_modem::rx_chain::{spawn_rx_thread, RxControl, RxStatus};
use ham_modem::tx_chain::{spawn_tx_thread, TxControl, TxStatus};

const TX_CONTROL_TOPIC: &str = "bpsk/tx/control";
const TX_STATUS_TOPIC: &str = "bpsk/tx/status";
const RX_CONTROL_TOPIC: &str = "bpsk/rx/control";
const RX_STATUS_TOPIC: &str = "bpsk/rx/status";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let broker_host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let broker_port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1883);
    let initial_bit_rate: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000.0);

    println!("BPSK BERT test node");
    println!("  MQTT broker:       {broker_host}:{broker_port}");
    println!("  Initial bit rate:  {initial_bit_rate} bps");
    println!("  TX control topic:  {TX_CONTROL_TOPIC}  (QoS1)");
    println!("  TX status topic:   {TX_STATUS_TOPIC}   (QoS0, retained)");
    println!("  RX control topic:  {RX_CONTROL_TOPIC}  (QoS1)");
    println!("  RX status topic:   {RX_STATUS_TOPIC}   (QoS0, retained)");
    println!();
    println!("Watch everything:");
    println!("  mosquitto_sub -h {broker_host} -p {broker_port} -t 'bpsk/#' -v");
    println!("Example commands:");
    println!("  mosquitto_pub -h {broker_host} -p {broker_port} -t {TX_CONTROL_TOPIC} -m '{{\"Prbs\":\"InjectError\"}}'");
    println!("  mosquitto_pub -h {broker_host} -p {broker_port} -t {TX_CONTROL_TOPIC} -m '{{\"Prbs\":\"Reset\"}}'");
    println!(
        "  mosquitto_pub -h {broker_host} -p {broker_port} -t {TX_CONTROL_TOPIC} -m '{{\"Throttle\":{{\"SetRate\":{{\"items_per_second\":5000.0}}}}}}'"
    );
    println!("  mosquitto_pub -h {broker_host} -p {broker_port} -t {RX_CONTROL_TOPIC} -m '{{\"Prbs\":\"ResetStats\"}}'");
    println!();

    // --- internal BERT loopback (no hardware yet) ---
    let (bit_tx, bit_rx) = bounded::<u8>(256);
    let (tx_control_tx, tx_control_rx) = unbounded::<TxControl>();
    let (tx_status_tx, tx_status_rx) = unbounded::<TxStatus>();
    let (rx_control_tx, rx_control_rx) = unbounded::<RxControl>();
    let (rx_status_tx, rx_status_rx) = unbounded::<RxStatus>();

    let _tx_handle = spawn_tx_thread(
        PrbsPattern::Pn15,
        initial_bit_rate,
        bit_tx,
        tx_control_rx,
        tx_status_tx,
        Duration::from_secs(1),
    );
    let _rx_handle = spawn_rx_thread(
        PrbsPattern::Pn15,
        bit_rx,
        rx_control_rx,
        rx_status_tx,
        Duration::from_secs(1),
    );

    // --- MQTT bridge ---
    let mut mqtt_options = MqttOptions::new("bpsk_bert_node", broker_host, broker_port);
    mqtt_options.set_keep_alive(Duration::from_secs(5));
    let (mqtt_client, mut mqtt_connection) = Client::new(mqtt_options, 64);

    mqtt_client
        .subscribe(TX_CONTROL_TOPIC, QoS::AtLeastOnce)
        .expect("subscribe to TX control topic");
    mqtt_client
        .subscribe(RX_CONTROL_TOPIC, QoS::AtLeastOnce)
        .expect("subscribe to RX control topic");

    // Thread 1: drain incoming MQTT control messages, forward into the
    // internal crossbeam control channels.
    {
        let tx_control_tx = tx_control_tx.clone();
        let rx_control_tx = rx_control_tx.clone();
        thread::spawn(move || {
            for notification in mqtt_connection.iter() {
                match notification {
                    Ok(Event::Incoming(Packet::Publish(publish))) => {
                        let payload = String::from_utf8_lossy(&publish.payload);
                        if publish.topic == TX_CONTROL_TOPIC {
                            match serde_json::from_str::<TxControl>(&payload) {
                                Ok(cmd) => {
                                    println!("[tx control] {payload} -> {cmd:?}");
                                    let _ = tx_control_tx.send(cmd);
                                }
                                Err(e) => eprintln!("[tx control] bad JSON '{payload}': {e}"),
                            }
                        } else if publish.topic == RX_CONTROL_TOPIC {
                            match serde_json::from_str::<RxControl>(&payload) {
                                Ok(cmd) => {
                                    println!("[rx control] {payload} -> {cmd:?}");
                                    let _ = rx_control_tx.send(cmd);
                                }
                                Err(e) => eprintln!("[rx control] bad JSON '{payload}': {e}"),
                            }
                        }
                    }
                    Ok(_) => {} // other MQTT events (ConnAck, PingResp, ...) - ignore
                    Err(e) => {
                        eprintln!("[mqtt] connection error: {e}");
                    }
                }
            }
        });
    }

    // Main thread: drain internal status channels, publish each to MQTT.
    loop {
        select! {
            recv(tx_status_rx) -> status => {
                if let Ok(status) = status {
                    let json = serde_json::to_string(&status).expect("serialize TxStatus");
                    let _ = mqtt_client.publish(TX_STATUS_TOPIC, QoS::AtMostOnce, true, json);
                }
            }
            recv(rx_status_rx) -> status => {
                if let Ok(status) = status {
                    let json = serde_json::to_string(&status).expect("serialize RxStatus");
                    let _ = mqtt_client.publish(RX_STATUS_TOPIC, QoS::AtMostOnce, true, json);
                }
            }
        }
    }
}
