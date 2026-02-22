/*
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-meduim.bin

*/

use reqwest::blocking::Client;
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::app::ModelType;

pub fn ensure_model_downloaded(model_type: ModelType) -> Result<PathBuf, anyhow::Error> {
    let model_path = model_type.model_path(); // assumes you have this method
    if model_path.exists() {
        println!("Transcription model present");
        return Ok(model_path);
    }

    // Create models directory if needed
    if let Some(parent) = model_path.parent() {
        println!("Creating Models directory in project root");
        fs::create_dir_all(parent)?;
    }

    let url = model_type.download_url(); // also defined in ModelType
    println!("Downloading model from {} ...", url);

    let client = Client::new();
    let response = client.get(&url).send()?;
    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded = 0;
    let mut file = fs::File::create(&model_path)?;

    let mut stream = response;
    let mut buffer = [0; 8192];
    loop {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        if total_size > 0 {
            let percent = (downloaded as f64 / total_size as f64) * 100.0;
            println!("Download progress: {:.1}%", percent);
        }
    }

    println!("Model downloaded to {:?}", model_path);
    Ok(model_path)
}

pub fn load_wav(wav_path: String) -> Result<Vec<f32>, hound::Error> {
    println!("\n Trying to open wav file with hound \n");
    let mut wav = hound::WavReader::open(wav_path)?;

    let spec = wav.spec();
    println!("\n SPEC {:?}", spec.clone());

    let mut samples_orig: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => wav
                .samples::<i16>()
                .map(|s| s.map(|x| x as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
            24 => wav
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 / 8388608.0))
                .collect::<Result<Vec<_>, _>>()?,
            32 => wav
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 / 2147483648.0))
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(hound::Error::InvalidSampleFormat),
        },
    };

    if spec.channels == 2 {
        println!("Converting to mono");
        samples_orig = stereo_to_mono(&samples_orig);
    }

    if spec.sample_rate != 16000 {
        println!("Resampling to 16000 Hz");
        samples_orig = audio_resample(&samples_orig, spec.sample_rate, 16000, 1);
    }

    println!("Wave file is now open and in correct format");

    Ok(samples_orig)
}

pub fn transcribe(
    model_type: ModelType,
    transcription_lang: String,
    samples: Vec<f32>,
    input_wav_path: &Path, // new parameter
    output_dir: &Path,
) -> Result<(), anyhow::Error> {
    let model_path = ensure_model_downloaded(model_type)?;
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Model path contains invalid UTF-8: {:?}", model_path))?;
    // load a context and model
    let ctx = WhisperContext::new_with_params(&model_path_str, WhisperContextParameters::default())
        .expect("failed to load model");
    // create a state attached to the model
    let mut state = ctx.create_state().expect("failed to create state");

    // the sampling strategy will determine how accurate your final output is going to be
    // typically BeamSearch is more accurate at the cost of significantly increased CPU time
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        // whisper.cpp defaults to a beam size of 5, a reasonable default
        beam_size: 5,
        // this parameter is currently unused but defaults to -1.0
        patience: -1.0,
    });

    println!("\n Model loaded fine ------------ \n");

    // and set the language to translate to as swedish
    params.set_language(Some(&transcription_lang));

    // we also explicitly disable anything that prints to stdout
    // despite all of this you will still get things printing to stdout,
    // be prepared to deal with it
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    println!("\nStarting transcription \n");

    state
        .full(params, &samples[..])
        .expect("failed to run model");

    let mut full_transcript: Vec<String> = Vec::new();

    // fetch the results
    for segment in state.as_iter() {
        if let Ok(text) = segment.to_str() {
            full_transcript.push(text.to_string());

            println!(
                "[{} - {}]: {}",
                // these timestamps are in centiseconds (10s of milliseconds)
                segment.start_timestamp(),
                segment.end_timestamp(),
                // this default Display implementation will result in any invalid UTF-8
                // being converted into the Unicode replacement character, U+FFFD
                segment
            );
        } else {
            eprintln!("Failed to store trancript");
        }
    }

    let transcript = full_transcript.join(" ");

    let stem = input_wav_path
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("Input file has no name: {:?}", input_wav_path))?;
    let transcript_filename = format!("{}_transcript.txt", stem.to_string_lossy());
    let transcript_path = output_dir.join(transcript_filename);

    // Save transcript
    let mut transcript_file = File::create(transcript_path)?;
    transcript_file.write_all(transcript.as_bytes())?;

    Ok(())
}

pub fn audio_resample(
    data: &[f32],
    sample_rate0: u32,
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    use samplerate::{ConverterType, convert};
    convert(
        sample_rate0,
        sample_rate,
        channels as _,
        ConverterType::SincBestQuality,
        data,
    )
    .unwrap_or_default()
}

pub fn stereo_to_mono(stereo_data: &[f32]) -> Vec<f32> {
    // Ensure the input data length is even (it should be if it's valid stereo data)
    assert_eq!(
        stereo_data.len() % 2,
        0,
        "Stereo data length should be even."
    );

    let mut mono_data = Vec::with_capacity(stereo_data.len() / 2);

    // Iterate over stereo data in steps of 2 (one stereo sample pair at a time)
    for chunk in stereo_data.chunks_exact(2) {
        // Calculate the average of the two channels
        let average = (chunk[0] + chunk[1]) / 2.0;
        mono_data.push(average);
    }

    mono_data
}
