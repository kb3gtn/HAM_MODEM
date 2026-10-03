pub mod agc;
pub mod ax25_signal;
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

/// Common control/status interface implemented by every PHY block.
pub trait Block {
    type Control;
    type Status: Clone;

    fn handle_control(&mut self, cmd: Self::Control);
    fn status(&self) -> Self::Status;
}
