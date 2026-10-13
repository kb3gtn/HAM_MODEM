//! SDR backends. Each backend turns a `RadioConfig` into a pair of boxed
//! `TxRadio` / `RxRadio` objects (the traits the engines in `modem` run on).
//! Backends are Cargo features so the DSP core builds without any SDR library.

use crate::modem::{RxRadio, TxRadio};

#[cfg(feature = "bladerf")]
pub mod bladerf;
#[cfg(feature = "soapy")]
pub mod soapy;

/// Everything a backend needs to bring the radio up. Frequencies are the LO
/// (the DSP shift is applied in software); the sample rate and analog
/// bandwidth come from `params::hardware_sample_rate_for` / `analog_bandwidth_hz`.
#[derive(Debug, Clone)]
pub struct RadioConfig {
    pub tx_enabled: bool,
    pub rx_enabled: bool,
    pub tx_freq_hz: u64,
    pub rx_freq_hz: u64,
    pub tx_gain_db: i32,
    pub rx_gain_db: i32,
    pub sample_rate_hz: u32,
    pub analog_bandwidth_hz: u32,
    /// SoapySDR args string selecting the device (e.g. `driver=bladerf`).
    pub device_args: String,
    /// 0-indexed port numbers for backends that choose them (SoapySDR).
    pub tx_channel: usize,
    pub rx_channel: usize,
    /// Backend-specific internal loopback mode ("none" for a real RF path).
    pub loopback: String,
}

/// The opened radios; a half that was not enabled is `None`.
pub struct Radios {
    pub tx: Option<Box<dyn TxRadio + Send>>,
    pub rx: Option<Box<dyn RxRadio + Send>>,
}

/// Names of the backends compiled into this build.
pub fn available_backends() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(feature = "bladerf")]
    v.push("bladerf");
    #[cfg(feature = "soapy")]
    v.push("soapy");
    v
}

/// Open the named backend (`"bladerf"`, ...).
#[cfg_attr(
    not(any(feature = "bladerf", feature = "soapy")),
    allow(unused_variables)
)]
pub fn open(backend: &str, cfg: &RadioConfig) -> Result<Radios, String> {
    match backend {
        #[cfg(feature = "bladerf")]
        "bladerf" => bladerf::open(cfg),
        #[cfg(feature = "soapy")]
        "soapy" => soapy::open(cfg),
        other => Err(format!(
            "radio backend '{other}' is unknown or not compiled in (available: {})",
            available_backends().join(", ")
        )),
    }
}
