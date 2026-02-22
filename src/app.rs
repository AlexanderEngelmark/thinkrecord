use rand::{RngExt, distr::Alphanumeric};
use std::env;
use std::path::PathBuf;

use eframe::egui;
use egui_file_dialog::FileDialog;

use crate::recorder::{GRATITUDE_TIME, INSTRUCTION_TIME, PAUSE_TIME, Recorder, TALK_TIME};
use crate::subject;
use crate::transcribe::{load_wav, transcribe};
use std::sync::mpsc::{self, Receiver, Sender};

/// derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    picked_out_directory: Option<PathBuf>,
    subject_label: String,
    session_label: String,
    perturbation_1_id: String,
    perturbation_2_id: String,
    #[serde(skip)] // This how to opt-out of serialization of a field
    out_file_dialog: FileDialog,
    #[serde(skip)]
    transcribe_file_dialog: FileDialog,
    file_to_transcribe: Option<PathBuf>,
    transcription_model: ModelType,
    transcription_language: TranscriptionLang,
    #[serde(skip)]
    transcribed_text: Option<Vec<String>>,
    #[serde(skip)]
    experiment_state: Option<ExperimentState>,
    #[serde(skip)]
    msg_sender: Sender<ExperimentMessage>,
    #[serde(skip)]
    msg_receiver: Receiver<ExperimentMessage>,
    #[serde(skip)]
    runtime: tokio::runtime::Runtime,
}

#[derive(PartialEq)]
enum ExperimentPhase {
    Idle,
    Instructions,
    Recording { block: usize },
    Pause,
    Gratitude,
}

struct ExperimentState {
    phase: ExperimentPhase,
    phase_start: std::time::Instant,
    block_count: usize, // total blocks done so far
}

enum ExperimentMessage {
    RecordingDone(usize), // block index that finished
    RecordingFailed(String),
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum ModelType {
    Tiny,
    Base,
    Small,
    Medium,
}

impl ModelType {
    /// Returns the filename of the model (e.g., "ggml-tiny.bin")
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
        }
    }

    /// Returns the full path to the model file inside the `models/` directory.
    pub fn model_path(&self) -> PathBuf {
        env::current_dir()
            .unwrap()
            .join("models")
            .join(self.filename())
    }

    /// Returns the download URL for the model.
    pub fn download_url(&self) -> String {
        let base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
        format!("{}{}", base, self.filename())
    }

    /// (Optional) Keep for backward compatibility if needed.
    /// Converts the path to a string using `/` separators (may not be portable).
    pub fn to_path_string(&self) -> String {
        self.model_path().to_string_lossy().to_string()
    }
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum TranscriptionLang {
    English,
    Swedish,
}

impl TranscriptionLang {
    fn lang_to_string(&self) -> String {
        match self {
            Self::English => "en".to_string(),
            Self::Swedish => "sv".to_string(),
        }
    }
}

impl Default for TemplateApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            subject_label: "None".to_owned(),
            session_label: "None".to_owned(),
            perturbation_1_id: "None".to_owned(),
            perturbation_2_id: "None".to_owned(),
            file_to_transcribe: None,
            transcribe_file_dialog: FileDialog::new(),
            out_file_dialog: FileDialog::new(),
            picked_out_directory: None,
            transcription_model: ModelType::Tiny,
            transcription_language: TranscriptionLang::Swedish,
            transcribed_text: None,
            experiment_state: None,
            msg_sender: tx,
            msg_receiver: rx,
            runtime: tokio::runtime::Runtime::new().unwrap(),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        }
    }

    fn advance_phase(&mut self) {
        // Determine the next phase and whether to clear the experiment state
        let (next_phase, next_start, should_clear) = {
            let state = match &mut self.experiment_state {
                Some(s) => s,
                None => return,
            };
            let now = std::time::Instant::now();

            match state.phase {
                ExperimentPhase::Idle => (ExperimentPhase::Idle, now, true),

                ExperimentPhase::Instructions => {
                    // Start first recording block (block 0)
                    (ExperimentPhase::Recording { block: 0 }, now, false)
                }

                ExperimentPhase::Recording { block } => {
                    if block == 2 {
                        // Last block of first set (blocks 0,1,2) → Gratitude
                        (ExperimentPhase::Gratitude, now, false)
                    } else if block == 8 {
                        // Last block of second set (blocks 3..8) → finish
                        (ExperimentPhase::Idle, now, true)
                    } else {
                        // Otherwise go to pause
                        (ExperimentPhase::Pause, now, false)
                    }
                }

                ExperimentPhase::Pause => {
                    let next_block = state.block_count; // already incremented after previous recording
                    (ExperimentPhase::Recording { block: next_block }, now, false)
                }

                ExperimentPhase::Gratitude => (ExperimentPhase::Recording { block: 3 }, now, false),
            }
        };

        // Apply the transition
        if should_clear {
            self.experiment_state = None;
        } else if let Some(state) = &mut self.experiment_state {
            state.phase = next_phase; // next_phase is moved here
            state.phase_start = next_start;

            // Now check the newly assigned phase
            if let ExperimentPhase::Recording { block } = state.phase {
                self.spawn_recording(block);
            }
        }
    }

    fn spawn_recording(&mut self, block: usize) {
        let btype = if block < 3 { "baseline" } else { "gratitude" };

        let path = self
            .picked_out_directory
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap())
            .join(format!(
                "Recording_Subject_{}_Session_{}_block_{}_{}.wav",
                self.subject_label, self.session_label, block, btype
            ))
            .to_string_lossy()
            .to_string();

        let tx = self.msg_sender.clone();
        // Spawn a blocking task (record_to_file is blocking)
        self.runtime
            .spawn_blocking(move || match Recorder::record_to_file(&path, TALK_TIME) {
                Ok(()) => {
                    tx.send(ExperimentMessage::RecordingDone(block))
                        .unwrap_or_default();
                }
                Err(e) => {
                    tx.send(ExperimentMessage::RecordingFailed(format!("{:?}", e)))
                        .unwrap_or_default();
                }
            });
    }

    fn render_participant_view(&mut self, ui: &mut egui::Ui) {
        // Fill the background (optional, e.g., black for better contrast)
        ui.painter()
            .rect_filled(ui.max_rect(), 0.0, egui::Color32::BLACK);

        if let Some(state) = &self.experiment_state {
            match state.phase {
                ExperimentPhase::Pause => {
                    // Only the fixation cross, perfectly centered
                    ui.centered_and_justified(|ui| {
                        ui.painter().text(
                            ui.max_rect().center(),
                            egui::Align2::CENTER_CENTER,
                            "+",
                            egui::FontId::proportional(150.0),
                            egui::Color32::WHITE, // use white on black background
                        );
                    });
                }

                _ => {
                    // For other phases, show timer and phase-specific content
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);

                        // Timer (remaining seconds)
                        let elapsed = state.phase_start.elapsed();
                        let total_duration = match state.phase {
                            ExperimentPhase::Instructions => INSTRUCTION_TIME,
                            ExperimentPhase::Recording { .. } => {
                                std::time::Duration::from_secs(TALK_TIME)
                            }
                            ExperimentPhase::Gratitude => GRATITUDE_TIME,
                            _ => unreachable!(),
                        };
                        let remaining = total_duration.saturating_sub(elapsed);
                        ui.label(
                            egui::RichText::new(format!("{} sekunder", remaining.as_secs()))
                                .size(80.0)
                                .strong()
                                .color(egui::Color32::WHITE),
                        );

                        ui.add_space(40.0);

                        // Block number (only during recording)
                        if let ExperimentPhase::Recording { block } = state.phase {
                            ui.label(
                                egui::RichText::new(format!("Omgång {} av 9", block + 1))
                                    .size(50.0)
                                    .color(egui::Color32::WHITE),
                            );
                            ui.add_space(30.0);
                        }

                        // Phase-specific text
                        match state.phase {
                            ExperimentPhase::Instructions => {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(
                                            "Instruktion: \n\n Om 120 sekunder ska du beskriva vad du tänker på och känner just nu genom att prata fritt. \n\n
                                             Efter detta kommer ett fixationskors att visas och då kan du pausa. \n\n
                                             Denna sekvens kommer sedan repeteras två gånger till. \n\n
                                             Efter detta kommer du få göra en kort övning och sedan kommer sekvensen att upprepas sex gånger."
                                        )
                                        .size(35.0)
                                        .color(egui::Color32::WHITE)
                                    )
                                    .wrap()
                                );
                            }
                            ExperimentPhase::Gratitude => {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(
                                            "Tänk på någonting som du är tacksam för"
                                        )
                                        .size(35.0)
                                        .color(egui::Color32::WHITE)
                                    )
                                    .wrap(),
                                );
                            }
                            _ => {}
                        }
                    });
                }
            }
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // Process messages from background tasks
        while let Ok(msg) = self.msg_receiver.try_recv() {
            match msg {
                ExperimentMessage::RecordingDone(block) => {
                    if let Some(state) = &mut self.experiment_state {
                        if let ExperimentPhase::Recording {
                            block: current_block,
                        } = state.phase
                        {
                            if current_block == block {
                                state.block_count += 1;
                                self.advance_phase();
                                ctx.request_repaint();
                            } else {
                                eprintln!(
                                    "Received RecordingDone for block {} but current phase is block {}",
                                    block, current_block
                                );
                            }
                        } else {
                            eprintln!("Received RecordingDone but not in recording phase");
                        }
                    }
                }
                ExperimentMessage::RecordingFailed(err) => {
                    eprintln!("Recording failed: {}", err);
                    self.experiment_state = None;
                    ctx.request_repaint();
                }
            }
        }

        // Handle experiment timing if active
        if let Some(state) = &mut self.experiment_state {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(state.phase_start);
            let phase_duration = match state.phase {
                ExperimentPhase::Instructions => INSTRUCTION_TIME,
                ExperimentPhase::Recording { .. } => std::time::Duration::from_secs(TALK_TIME),
                ExperimentPhase::Pause => PAUSE_TIME,
                ExperimentPhase::Gratitude => GRATITUDE_TIME,
                ExperimentPhase::Idle => unreachable!(),
            };

            if elapsed >= phase_duration {
                match state.phase {
                    ExperimentPhase::Recording { .. } => {
                        eprintln!(
                            "Warning: Recording phase timed out – waiting for completion message."
                        );
                    }
                    _ => {
                        self.advance_phase();
                        // No need to repaint here – we'll do it below
                    }
                }
            }
            // Always request a repaint while experiment is running
            ctx.request_repaint();
        }

        if self.experiment_state.is_some() {
            // Participant view: full screen, no panels
            egui::CentralPanel::default().show(ctx, |ui| {
                self.render_participant_view(ui);
            });
        } else {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                // The top panel is often a good place for a menu bar:

                egui::MenuBar::new().ui(ui, |ui| {
                    // NOTE: no File->Quit on web pages!
                    let is_web = cfg!(target_arch = "wasm32");
                    if !is_web {
                        ui.menu_button("File", |ui| {
                            if ui.button("Quit").clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                        ui.add_space(16.0);
                    }

                    egui::widgets::global_theme_preference_buttons(ui);
                });
            });

            egui::SidePanel::right("right_panel")
                .min_width(300.0)
                .show(ctx, |ui| {
                    ui.separator();

                    ui.heading("Right Panel");
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                // The central panel the region left after adding TopPanel's and SidePanel's
                ui.heading("Think-aloud recorder");

                if ui.button("Pick output directory").clicked() {
                    self.out_file_dialog.pick_directory();
                }

                ui.label(format!(
                    "Picked directory: {:?}",
                    self.picked_out_directory
                        .clone()
                        .unwrap_or_default()
                        .into_os_string()
                ));

                if let Some(path) = self.out_file_dialog.update(ctx).picked() {
                    self.picked_out_directory = Some(path.to_path_buf());
                }

                ui.heading("Session metadata");
                ui.separator();

                if ui.button("Generate subject ID").clicked() {
                    let s: String = rand::rng()
                        .sample_iter(&Alphanumeric)
                        .take(7)
                        .map(char::from)
                        .collect();
                    self.subject_label = s;
                }

                ui.horizontal(|ui| {
                    ui.label("Subject: ");
                    ui.text_edit_singleline(&mut self.subject_label);
                });

                if ui.button("Generate session ID").clicked() {
                    let s: String = rand::rng()
                        .sample_iter(&Alphanumeric)
                        .take(7)
                        .map(char::from)
                        .collect();
                    self.session_label = s;
                }

                ui.horizontal(|ui| {
                    ui.label("Session: ");
                    ui.text_edit_singleline(&mut self.session_label);
                });

                ui.horizontal(|ui| {
                    ui.label("Perturbation 1: ");
                    ui.text_edit_singleline(&mut self.perturbation_1_id);
                });

                ui.horizontal(|ui| {
                    ui.label("Perturbation 2: ");
                    ui.text_edit_singleline(&mut self.perturbation_2_id);
                });

                ui.separator();

                ui.heading("Start recording");

                ui.separator();

                if ui.button("Start recording to local .wav file").clicked() {
                    // Save metadata first
                    let wav_save_name = format!(
                        "{}/Recording_Subject_{}_Session_{}.wav",
                        self.picked_out_directory
                            .clone()
                            .unwrap_or_else(|| std::env::current_dir().unwrap())
                            .to_string_lossy(),
                        self.subject_label,
                        self.session_label
                    );
                    if let Err(e) = subject::save_session_metadata(
                        self.subject_label.clone(),
                        self.session_label.clone(),
                        self.perturbation_1_id.clone(),
                        self.perturbation_2_id.clone(),
                        TALK_TIME,
                        wav_save_name.clone(),
                    ) {
                        eprintln!("Saving of metadata failed: {}", e);
                    }

                    // Initialize experiment state
                    self.experiment_state = Some(ExperimentState {
                        phase: ExperimentPhase::Instructions,
                        phase_start: std::time::Instant::now(),
                        block_count: 0,
                    });

                    ctx.request_repaint();
                }

                ui.separator();

                ui.heading("Off-line transcription of .wav file");

                if ui.button("Pick .wav file to transcribe").clicked() {
                    // Open the file dialog to pick a file.
                    self.transcribe_file_dialog.pick_file();
                }

                ui.label(format!(
                    "Picked file: {:?} to transcribe",
                    self.file_to_transcribe
                        .clone()
                        .unwrap_or_default()
                        .into_os_string()
                ));

                // Update the dialog
                self.transcribe_file_dialog.update(ctx);
                // Check if the user picked a file.
                if let Some(path) = self.transcribe_file_dialog.take_picked() {
                    self.file_to_transcribe = Some(path.to_path_buf());
                }

                egui::ComboBox::from_label(
                    "Select language for transcription (for on-line and off-line transcription)",
                )
                .selected_text(format!("{:?}", self.transcription_language))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.transcription_language,
                        TranscriptionLang::Swedish,
                        "Swedish",
                    );
                    ui.selectable_value(
                        &mut self.transcription_language,
                        TranscriptionLang::English,
                        "English",
                    );
                });

                egui::ComboBox::from_label(
                    "Select model for text transcription (for on-line and off-line transcription)",
                )
                .selected_text(format!("{:?}", self.transcription_model))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.transcription_model, ModelType::Tiny, "tiny");
                    ui.selectable_value(&mut self.transcription_model, ModelType::Base, "base");
                    ui.selectable_value(&mut self.transcription_model, ModelType::Small, "small");
                    ui.selectable_value(&mut self.transcription_model, ModelType::Medium, "medium");
                });

                if ui.button("Transcribe local .wav file").clicked() {
                    // Get the input WAV path and output directory
                    let input_path = self.file_to_transcribe.clone().unwrap();
                    let output_dir = self
                        .picked_out_directory
                        .clone()
                        .unwrap_or_else(|| std::env::current_dir().unwrap());

                    // Load samples from the WAV file
                    match load_wav(input_path.to_string_lossy().to_string()) {
                        Ok(samples) => {
                            println!("OK TO TRANSCRIBE");
                            match transcribe(
                                self.transcription_model,
                                self.transcription_language.lang_to_string(),
                                samples,
                                &input_path, // pass as &Path
                                &output_dir, // pass as &Path
                            ) {
                                Ok(()) => println!("Transcription successful"),
                                Err(e) => eprintln!("Transcription failed: {}", e),
                            }
                        }
                        Err(e) => eprintln!("WAV loading failed: {}", e),
                    }
                }

                ui.separator();
                ui.heading("Output:");
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {});
                ui.separator();

                // ui.add(egui::github_link_file!(
                //     "https://github.com/emilk/eframe_template/blob/main/",
                //     "Source code."
                // ));

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    powered_by_egui_and_eframe(ui);
                    egui::warn_if_debug_build(ui);
                });
            });
        }
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
