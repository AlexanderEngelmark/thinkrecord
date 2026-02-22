use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use crate::SessionMetadata;

pub fn save_session_metadata(
    subject_id: String,
    session_id: String,
    perturbation_1_id: String,
    perturbation_2_id: String,
    talk_time: u64,
    save_path: String,
) -> std::io::Result<()> {
    let session_metadata = SessionMetadata {
        subject_id: subject_id,
        session_id: session_id,
        perturbation_1_id: perturbation_1_id,
        perturbation_2_id: perturbation_2_id,
        talk_duration: talk_time,
    };

    // Serialize it to a JSON string.
    let j = serde_json::to_string(&session_metadata)?;

    // Print, write to a file, or send to an HTTP server.
    println!("Saving as {}", j);

    let full_save_path = format!("{}_{}", save_path.replace(".", "_"), "metadata.json");

    let file = File::create(full_save_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &j)?;
    writer.flush()?;

    Ok(())
}
