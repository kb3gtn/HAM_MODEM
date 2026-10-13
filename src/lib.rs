pub mod agc;
pub mod ax25_signal;
#[cfg(feature = "bladerf")]
pub mod capture;
pub mod carrier_recovery;
pub mod coarse_freq;
pub mod conv;
pub mod crc;
pub mod dc_block;
pub mod fec_bank;
pub mod hdlc;
pub mod kiss;
pub mod modem;
pub mod params;
pub mod prbs;
pub mod psd;
pub mod radio;
pub mod resampler;
pub mod rrc;
pub mod rx_bert;
pub mod rx_chain;
pub mod rx_signal;
pub mod scrambler;
pub mod symbol_map;
pub mod throttle;
pub mod timing_recovery;
pub mod tx_bert;
pub mod tx_chain;
pub mod tx_signal;

/// One hardware-rate IQ sample: signed 16-bit I and Q (SC16, Q11 full scale).
/// The same layout every supported SDR backend streams, so no conversion is
/// needed at the radio edge.
pub type Iq16 = num_complex::Complex<i16>;

/// Common control/status interface implemented by every PHY block.
pub trait Block {
    type Control;
    type Status: Clone;

    fn handle_control(&mut self, cmd: Self::Control);
    fn status(&self) -> Self::Status;
}
