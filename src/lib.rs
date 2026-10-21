use std::fmt;

pub const MAGIC: [u8; 8] = [b'V', b'L', b'A', b'C', 0x0D, 0x0A, 0x1A, 0x0A];
pub const VERSION: u8 = 1;
pub const MAX_CHANNELS: u8 = 2;
pub const MAX_SAMPLE_RATE: u32 = 192_000;

#[derive(Debug)]
pub enum VlacError {
    TooShort,
    BadMagic,
    UnsupportedVersion,
    UnsupportedFormat,
    ZstdError(String),
    CorruptData,
}

impl fmt::Display for VlacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "File too short"),
            Self::BadMagic => write!(f, "Invalid vlac magic bytes"),
            Self::UnsupportedVersion => write!(f, "Unsupported version"),
            Self::UnsupportedFormat => write!(f, "Unsupported sample rate or channels"),
            Self::ZstdError(e) => write!(f, "Zstd error: {}", e),
            Self::CorruptData => write!(f, "Corrupt data or dictionary mismatch"),
        }
    }
}

impl std::error::Error for VlacError {}

pub struct Header {
    pub sample_rate: u32,
    pub channels: u8,
    pub total_samples: u32,
    pub dict_hash: u32,
    pub compressed_size: u32,
}

impl Header {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(26);
        buf.extend_from_slice(&MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&self.sample_rate.to_le_bytes());
        buf.push(self.channels);
        buf.extend_from_slice(&self.total_samples.to_le_bytes());
        buf.extend_from_slice(&self.dict_hash.to_le_bytes());
        buf.extend_from_slice(&self.compressed_size.to_le_bytes());
        buf
    }

    pub fn decode(data: &[u8]) -> Result<(Self, usize), VlacError> {
        if data.len() < 26 {
            return Err(VlacError::TooShort);
        }
        if data[0..8] != MAGIC {
            return Err(VlacError::BadMagic);
        }
        if data[8] != VERSION {
            return Err(VlacError::UnsupportedVersion);
        }
        let sample_rate = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
        let channels = data[13];
        let total_samples = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);
        let dict_hash = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
        let compressed_size = u32::from_le_bytes([data[22], data[23], data[24], data[25]]);

        if channels == 0 || channels > MAX_CHANNELS || sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
            return Err(VlacError::UnsupportedFormat);
        }

        Ok((Self { sample_rate, channels, total_samples, dict_hash, compressed_size }, 26))
    }
}

pub fn hash_bytes(bytes: &[u8]) -> u32 {
    let mut h = 2166136261u32;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

pub fn encode(samples: &[i16], sample_rate: u32, channels: u8, dict: Option<&[u8]>) -> Result<Vec<u8>, VlacError> {
    let n = samples.len();
    let mut high_bytes = Vec::with_capacity(n * 2);
    let mut low_bytes = Vec::with_capacity(n * 2);
    
    let mut prev1 = 0i32;
    let mut prev2 = 0i32;
    let mut prev3 = 0i32;

    for &s in samples {
        let actual = s as i32;
        // Order 4 Predictor (using previous 3 samples to predict the next)
        let predicted = (3 * prev1) - (3 * prev2) + prev3;
        let residual = actual.wrapping_sub(predicted);
        let zz = zigzag_encode(residual);
        
        // High/Low Byte Split
        high_bytes.push((zz & 0xFF) as u8);
        high_bytes.push(((zz >> 8) & 0xFF) as u8);
        low_bytes.push(((zz >> 16) & 0xFF) as u8);
        low_bytes.push(((zz >> 24) & 0xFF) as u8);
        
        prev3 = prev2;
        prev2 = prev1;
        prev1 = actual;
    }

    // Concatenate High Bytes then Low Bytes
    let mut raw_bytes = high_bytes;
    raw_bytes.extend_from_slice(&low_bytes);

    let compressed = if let Some(d) = dict {
        let enc_dict = zstd::dict::EncoderDictionary::copy(d, 19);
        let mut enc = zstd::stream::Encoder::with_prepared_dictionary(Vec::new(), &enc_dict)
            .map_err(|e| VlacError::ZstdError(e.to_string()))?;
        std::io::copy(&mut &raw_bytes[..], &mut enc).map_err(|e| VlacError::ZstdError(e.to_string()))?;
        enc.finish().map_err(|e| VlacError::ZstdError(e.to_string()))?
    } else {
        zstd::bulk::compress(&raw_bytes, 19).map_err(|e| VlacError::ZstdError(e.to_string()))?
    };

    let dict_hash = dict.map(hash_bytes).unwrap_or(0);

    let header = Header {
        sample_rate,
        channels,
        total_samples: n as u32,
        dict_hash,
        compressed_size: compressed.len() as u32,
    };

    let mut out = header.encode();
    out.extend_from_slice(&compressed);
    Ok(out)
}

pub fn decode(data: &[u8], dict: Option<&[u8]>) -> Result<(Vec<i16>, u32, u8), VlacError> {
    let (header, offset) = Header::decode(data)?;

    if header.dict_hash != 0 {
        let dict_bytes = dict.ok_or(VlacError::CorruptData)?;
        if hash_bytes(dict_bytes) != header.dict_hash {
            return Err(VlacError::CorruptData);
        }
    }

    let payload = &data[offset..];
    if payload.len() < header.compressed_size as usize {
        return Err(VlacError::TooShort);
    }

    let sample_count = header.total_samples as usize;
    let expected_raw_size = sample_count * 4; // 2 bytes High + 2 bytes Low per sample
    
    let raw_bytes = if let Some(d) = dict {
        let dec_dict = zstd::dict::DecoderDictionary::copy(d);
        let mut dec = zstd::stream::Decoder::with_prepared_dictionary(payload, &dec_dict)
            .map_err(|e| VlacError::ZstdError(e.to_string()))?;
        let mut out = Vec::with_capacity(expected_raw_size);
        std::io::copy(&mut dec, &mut out).map_err(|e| VlacError::ZstdError(e.to_string()))?;
        out
    } else {
        zstd::bulk::decompress(payload, expected_raw_size).map_err(|e| VlacError::ZstdError(e.to_string()))?
    };

    if raw_bytes.len() != expected_raw_size {
        return Err(VlacError::CorruptData);
    }

    let mut samples = Vec::with_capacity(sample_count);
    let mut prev1 = 0i32;
    let mut prev2 = 0i32;
    let mut prev3 = 0i32;

    let high_bytes = &raw_bytes[0..sample_count * 2];
    let low_bytes = &raw_bytes[sample_count * 2..];

    for i in 0..sample_count {
        let h1 = high_bytes[i * 2] as u32;
        let h2 = high_bytes[i * 2 + 1] as u32;
        let l1 = low_bytes[i * 2] as u32;
        let l2 = low_bytes[i * 2 + 1] as u32;

        let zz = (l2 << 24) | (l1 << 16) | (h2 << 8) | h1;
        let residual = zigzag_decode(zz);
        
        // Order 4 Predictor
        let predicted = (3 * prev1) - (3 * prev2) + prev3;
        let actual = residual.wrapping_add(predicted);
        
        samples.push(actual.clamp(-32768, 32767) as i16);
        
        prev3 = prev2;
        prev2 = prev1;
        prev1 = actual;
    }

    Ok((samples, header.sample_rate, header.channels))
}
