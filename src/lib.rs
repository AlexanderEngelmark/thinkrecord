#![warn(clippy::all, rust_2018_idioms)]

use serde::{Deserialize, Serialize};

mod app;
pub use app::TemplateApp;
pub mod recorder;
pub mod subject;
pub mod transcribe;

#[derive(Serialize, Deserialize)]
pub struct SessionMetadata {
    pub subject_id: String,
    pub session_id: String,
    pub perturbation_1_id: String,
    pub perturbation_2_id: String,
    pub talk_duration: u64,
}
