use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "vlac", about = "Voice Lossless Audio Codec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Encode {
        #[arg(value_name = "INPUT_WAV")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT_VLAC")]
        output: PathBuf,
        #[arg(long, value_name = "DICT_FILE")]
        dict: Option<PathBuf>,
    },
    Decode {
        #[arg(value_name = "INPUT_VLAC")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT_WAV")]
        output: PathBuf,
        #[arg(long, value_name = "DICT_FILE")]
        dict: Option<PathBuf>,
    },
    Train {
        #[arg(value_name = "INPUT_DIR")]
        input_dir: PathBuf,
        #[arg(value_name = "OUTPUT_DICT")]
        output: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode { input, output, dict } => {
            let mut reader = hound::WavReader::open(&input)?;
            let spec = reader.spec();
            if spec.channels != 1 || spec.sample_rate > 192_000 {
                return Err("vlac only supports mono audio up to 192kHz".into());
            }
            
            let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
            
            let dict_bytes = if let Some(path) = dict {
                Some(fs::read(path)?)
            } else {
                None
            };

            let encoded = vlac::encode(&samples, spec.sample_rate, spec.channels as u8, dict_bytes.as_deref())?;
            fs::write(output, encoded)?;
            println!("Successfully encoded to vlac.");
        }
        Commands::Decode { input, output, dict } => {
            let data = fs::read(&input)?;
            
            let dict_bytes = if let Some(path) = dict {
                Some(fs::read(path)?)
            } else {
                None
            };

            let (samples, sample_rate, channels) = vlac::decode(&data, dict_bytes.as_deref())?;
            
            let spec = hound::WavSpec {
                channels: channels as u16,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(output, spec)?;
            for sample in samples {
                writer.write_sample(sample)?;
            }
            println!("Successfully decoded to WAV.");
        }
        Commands::Train { input_dir, output } => {
            let mut all_data = Vec::new();
            
            for entry in fs::read_dir(input_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("wav") {
                    let mut reader = match hound::WavReader::open(&path) {
                        Ok(r) => r,
                        Err(e) => {
                            println!("Skipping {}: {}", path.display(), e);
                            continue;
                        }
                    };
                    
                    let samples: Vec<i16> = reader.samples::<i16>().filter_map(|s| s.ok()).collect();
                    if samples.is_empty() { continue; }

                    let mut prev1 = 0i32;
                    let mut prev2 = 0i32;
                    let mut prev3 = 0i32;
                    
                    let mut high = Vec::with_capacity(samples.len() * 2);
                    let mut low = Vec::with_capacity(samples.len() * 2);

                    for &s in &samples {
                        let actual = s as i32;
                        let predicted = (3 * prev1) - (3 * prev2) + prev3;
                        let residual = actual.wrapping_sub(predicted);
                        let zz = ((residual << 1) ^ (residual >> 31)) as u32;
                        
                        high.push((zz & 0xFF) as u8);
                        high.push(((zz >> 8) & 0xFF) as u8);
                        low.push(((zz >> 16) & 0xFF) as u8);
                        low.push(((zz >> 24) & 0xFF) as u8);

                        prev3 = prev2;
                        prev2 = prev1;
                        prev1 = actual;
                    }
                    
                    all_data.extend_from_slice(&high);
                    all_data.extend_from_slice(&low);
                }
            }
            
            if all_data.is_empty() {
                return Err("No usable .wav files found.".into());
            }

            println!("Collected {} bytes of split residuals.", all_data.len());
            
            let dict_size = all_data.len().min(110_000);
            let raw_dict = &all_data[0..dict_size];
            
            fs::write(output, raw_dict)?;
            println!("Created {} byte raw dictionary.", dict_size);
        }
    }
    Ok(())
}
