//! AX.25's FCS: CRC-16/X-25 (poly 0x1021, reflected, init 0xFFFF, final
//! complement) - the standard HDLC frame check sequence.

/// Computes the FCS over `bytes`. For a transmitted frame, append the result
/// low-byte-first (`fcs & 0xFF`, then `fcs >> 8`) after the frame content.
pub fn ax25_fcs(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in bytes {
        crc ^= u16::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x8408;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC-16/X-25 check value for the ASCII string
    /// "123456789" - the usual way to confirm a CRC implementation matches
    /// the standard parameters (poly/init/refin/refout/xorout) rather than
    /// some other similar-looking variant.
    #[test]
    fn matches_the_standard_check_value() {
        assert_eq!(ax25_fcs(b"123456789"), 0x906E);
    }

    #[test]
    fn appending_the_fcs_and_rechecking_validates() {
        let data = b"a real AX.25-ish payload, arbitrary content";
        let fcs = ax25_fcs(data);
        let mut with_fcs = data.to_vec();
        with_fcs.push((fcs & 0xFF) as u8);
        with_fcs.push((fcs >> 8) as u8);

        let recomputed = ax25_fcs(&with_fcs[..data.len()]);
        assert_eq!(recomputed, fcs);
    }

    #[test]
    fn a_single_bit_flip_changes_the_fcs() {
        let data = b"detect this corruption please";
        let fcs_clean = ax25_fcs(data);
        let mut corrupted = data.to_vec();
        corrupted[5] ^= 0x01;
        let fcs_corrupted = ax25_fcs(&corrupted);
        assert_ne!(fcs_clean, fcs_corrupted);
    }
}
