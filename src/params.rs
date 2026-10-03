//! Modem parameters shared by every binary and test. The hardware sample
//! rate, RRC shape and oversampling are fixed; the symbol rate and the
//! modulation are runtime-selectable (see `modem::TxControlMsg` /
//! `RxControlMsg`), within the limits below, and the constants here are their
//! DEFAULTS.

use crate::symbol_map::Modulation;

/// Default symbol rate (symbols/second).
pub const SYMBOL_RATE_HZ: f64 = 16_000.0;
/// Lowest selectable symbol rate. Below this the RX decimation from the
/// 800 kHz hardware rate (SPS x rate / 800 kHz) becomes extreme.
pub const MIN_SYMBOL_RATE_HZ: f64 = 2_000.0;
/// Highest selectable symbol rate. At the high sample-rate profile
/// (`HIGH_HARDWARE_SAMPLE_RATE`) the DSP runs at 4 samples per symbol here.
/// There is no bandwidth cap (see `occupied_bandwidth_hz`).
pub const MAX_SYMBOL_RATE_HZ: f64 = 1_000_000.0;
/// Default modulation.
pub const DEFAULT_MODULATION: Modulation = Modulation::Psk8;
/// Bits carried per symbol at the default modulation (8PSK).
pub const BITS_PER_SYMBOL: usize = 3;
/// Raw physical-channel bit rate (coded bits on the air) at the defaults: 48 kbps.
pub const BIT_RATE_BPS: f64 = SYMBOL_RATE_HZ * BITS_PER_SYMBOL as f64;
/// Convolutional code rate (K=7, r=1/2): each data bit becomes two coded bits.
pub const CODE_RATE: f64 = 0.5;
/// Data bit rate after FEC (24 kbps at the defaults), before HDLC framing overhead.
pub const INFO_BIT_RATE_BPS: f64 = BIT_RATE_BPS * CODE_RATE;
/// RRC excess-bandwidth factor.
pub const RRC_ROLLOFF: f64 = 0.35;
/// RRC filter length in symbols.
pub const RRC_SPAN_SYMBOLS: usize = 8;
/// Symbol-domain samples per symbol at the standard rate (see `sps_for`).
pub const SPS: usize = 8;
/// The standard bladeRF IQ sample rate: used up to `LOW_PROFILE_MAX_SYMBOL_RATE_HZ`.
pub const HARDWARE_SAMPLE_RATE: u32 = 800_000;
/// The high-rate profile's bladeRF IQ sample rate.
pub const HIGH_HARDWARE_SAMPLE_RATE: u32 = 4_000_000;
/// Highest symbol rate served by the standard 800 kSPS profile at `SPS`
/// samples per symbol; above it the SDR runs at `HIGH_HARDWARE_SAMPLE_RATE`.
pub const LOW_PROFILE_MAX_SYMBOL_RATE_HZ: f64 = 100_000.0;
/// Fewest symbol-domain samples per symbol the timing recovery is run with.
pub const MIN_SPS: usize = 4;
/// Reference occupied-bandwidth mask for the DEFAULT symbol rate (25 kHz
/// channel). Used by the TX spectrum test only; the modem itself enforces no
/// bandwidth limit when the symbol rate is changed.
pub const MAX_OCCUPIED_BANDWIDTH_HZ: f64 = 25_000.0;
/// Theoretical RRC occupied bandwidth at the default rate: 21.6 kHz.
pub const OCCUPIED_BANDWIDTH_HZ: f64 = SYMBOL_RATE_HZ * (1.0 + RRC_ROLLOFF);

/// Theoretical RRC occupied bandwidth at `symbol_rate_hz`: rate x (1 + rolloff).
pub fn occupied_bandwidth_hz(symbol_rate_hz: f64) -> f64 {
    symbol_rate_hz * (1.0 + RRC_ROLLOFF)
}

/// Raw on-air bit rate for a modulation and symbol rate.
pub fn bit_rate_bps(modulation: Modulation, symbol_rate_hz: f64) -> f64 {
    symbol_rate_hz * modulation.bits_per_symbol() as f64
}

/// Data bit rate after FEC, before HDLC framing overhead.
pub fn info_bit_rate_bps(modulation: Modulation, symbol_rate_hz: f64) -> f64 {
    bit_rate_bps(modulation, symbol_rate_hz) * CODE_RATE
}

/// The SDR sample rate to run at for a symbol rate: the standard profile up to
/// `LOW_PROFILE_MAX_SYMBOL_RATE_HZ`, the high one above it.
pub fn hardware_sample_rate_for(symbol_rate_hz: f64) -> u32 {
    if symbol_rate_hz <= LOW_PROFILE_MAX_SYMBOL_RATE_HZ {
        HARDWARE_SAMPLE_RATE
    } else {
        HIGH_HARDWARE_SAMPLE_RATE
    }
}

/// Symbol-domain samples per symbol for a symbol rate at a hardware sample
/// rate: `SPS`, or as many as fit (at least `MIN_SPS`) when the hardware
/// rate is too low for that.
pub fn sps_for(symbol_rate_hz: f64, hardware_sample_rate: u32) -> usize {
    ((f64::from(hardware_sample_rate) / symbol_rate_hz).floor() as usize).clamp(MIN_SPS, SPS)
}

/// Analog (RF front-end filter) bandwidth for a symbol rate: the occupied
/// bandwidth with 20% margin, never below the bladeRF's 200 kHz minimum.
pub fn analog_bandwidth_hz(symbol_rate_hz: f64) -> u32 {
    (occupied_bandwidth_hz(symbol_rate_hz) * 1.2)
        .max(200_000.0)
        .ceil() as u32
}

/// IQ frames per radio write/read: about 10 ms of samples, in multiples of
/// 1024 (8192 at 800 kSPS).
pub fn chunk_frames_for(hardware_sample_rate: u32) -> usize {
    (hardware_sample_rate as usize / 100).next_multiple_of(1024)
}

const _: () = assert!(
    LOW_PROFILE_MAX_SYMBOL_RATE_HZ * SPS as f64 <= HARDWARE_SAMPLE_RATE as f64,
    "the standard profile cannot carry its maximum symbol rate at SPS samples per symbol"
);
const _: () = assert!(
    MAX_SYMBOL_RATE_HZ * MIN_SPS as f64 <= HIGH_HARDWARE_SAMPLE_RATE as f64,
    "the high profile cannot carry the maximum symbol rate at MIN_SPS samples per symbol"
);
