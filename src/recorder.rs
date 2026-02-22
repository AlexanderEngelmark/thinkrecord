use cpal::{
    DeviceId,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{
    str::FromStr,
    sync::{Arc, Mutex},
};

pub struct Recorder;

pub const INSTRUCTION_TIME: tokio::time::Duration = tokio::time::Duration::from_secs(120);
pub const PAUSE_TIME: tokio::time::Duration = tokio::time::Duration::from_secs(60);
pub const GRATITUDE_TIME: tokio::time::Duration = tokio::time::Duration::from_secs(180);

pub const TALK_TIME: u64 = 60;

#[derive(Debug)]
pub enum MicError {
    DeviceNotFound,
    StreamBuildFailed,
    Io(std::io::Error),
    Other(String),
}

impl From<std::io::Error> for MicError {
    fn from(e: std::io::Error) -> Self {
        MicError::Io(e)
    }
}

impl Recorder {
    async fn _record_one_session(path: &str) -> std::io::Result<()> {
        // Sleep for when instructions are given
        tokio::time::sleep(INSTRUCTION_TIME).await;

        for block_number in 0..2 {
            let full_save_path = format!("{}_block_{}", path, block_number,);

            match self::Recorder::record_to_file(&full_save_path, TALK_TIME) {
                Ok(()) => println!("Recording completed successfully!"),
                Err(MicError::DeviceNotFound) => eprintln!("No microphone found"),
                Err(MicError::Other(msg)) => eprintln!("Recording error: {}", msg),
                _ => eprintln!("Unknown error"),
            }

            tokio::time::sleep(PAUSE_TIME).await;
        }

        tokio::time::sleep(GRATITUDE_TIME).await;

        for block_number in 0..5 {
            let full_save_path = format!("{}_block_{}", path, block_number,);

            match self::Recorder::record_to_file(&full_save_path, TALK_TIME) {
                Ok(()) => println!("Recording completed successfully!"),
                Err(MicError::DeviceNotFound) => eprintln!("No microphone found"),
                Err(MicError::Other(msg)) => eprintln!("Recording error: {}", msg),
                _ => eprintln!("Unknown error"),
            }

            tokio::time::sleep(PAUSE_TIME).await;
        }

        Ok(())
    }

    pub fn record_to_file(path: &str, duration_sec: u64) -> Result<(), MicError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(MicError::DeviceNotFound)?;
        let config = device
            .default_input_config()
            .map_err(|e| MicError::Other(format!("Failed to get input config: {}", e)))?;

        let unknown_dev_id: &str = "Unknown";

        println!(
            "Recording with device: {}",
            device
                .id()
                .unwrap_or_else(|_| DeviceId::from_str(unknown_dev_id).unwrap())
        );
        println!("Input format: {:?}", config);

        let sample_format = config.sample_format();
        let config = config.into();

        match sample_format {
            cpal::SampleFormat::F32 => Self::record::<f32>(&device, &config, path, duration_sec),
            cpal::SampleFormat::I16 => Self::record::<i16>(&device, &config, path, duration_sec),
            cpal::SampleFormat::I32 => Self::record::<i32>(&device, &config, path, duration_sec),
            cpal::SampleFormat::I8 => Self::record::<i8>(&device, &config, path, duration_sec),
            _ => {
                return Err(MicError::Other(format!(
                    "Unsupported format: {:?}",
                    sample_format
                )));
            }
        }
    }

    fn record<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        path: &str,
        duration_sec: u64,
    ) -> Result<(), MicError>
    where
        T: cpal::Sample + hound::Sample + Send + cpal::SizedSample + 'static,
    {
        let spec = hound::WavSpec {
            channels: config.channels,
            sample_rate: config.sample_rate,
            bits_per_sample: std::mem::size_of::<T>() as u16 * 8,
            sample_format: if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let writer = hound::WavWriter::create(path, spec)
            .map_err(|e| MicError::Other(format!("Failed to create WAV writer: {}", e)))?;
        let writer = Arc::new(Mutex::new(Some(writer)));
        let writer_clone = Arc::clone(&writer);

        let err_fn = |err| eprintln!("Stream error: {}", err);

        let stream = device
            .build_input_stream(
                config,
                move |data: &[T], _| {
                    let mut writer_lock = writer_clone.lock().unwrap();
                    if let Some(w) = writer_lock.as_mut() {
                        for &sample in data {
                            if let Err(e) = w.write_sample(sample) {
                                eprintln!("Error writing sample: {}", e);
                            }
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| MicError::Other(format!("Failed to build input stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| MicError::Other(format!("Failed to start stream: {}", e)))?;

        std::thread::sleep(std::time::Duration::from_secs(duration_sec));

        // Stop the stream and finalize the writer
        drop(stream);

        writer
            .lock()
            .unwrap()
            .take()
            .map(|w| w.finalize())
            .transpose()
            .map_err(|e| MicError::Other(format!("Failed to finalize WAV file: {}", e)))?;

        Ok(())
    }
}
