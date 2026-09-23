//! Ties every verified piece together into one persistent-state streaming
//! engine: `StreamingStft` -> band-split/sub-band norm -> cached speaker
//! gate -> `BsrnnSeparator` (stateful) -> band-masker -> `StreamingIstft`.
//! A direct structural port of the already-verified Python
//! `poc/real_tse/src/streaming_engine.py::StreamingTseEngine`.

use crate::separator::{BsrnnSeparator, RepeatBlockState};
use crate::speaker_fusion::project_embedding;
use crate::stft::{StreamingIstft, StreamingStft};

pub struct StreamingTseEngine {
    stft: StreamingStft,
    istft: StreamingIstft,
    separator: BsrnnSeparator,
    repeat_block_state: Vec<RepeatBlockState>,
    gate: [f32; 128],
}

impl StreamingTseEngine {
    /// `target_embedding`: the real 192-dim ECAPA embedding for the
    /// enrolled target speaker (from `SpeakerEncoderOnnx::embed_from_fbank`),
    /// computed ONCE per enrollment - never recomputed per frame. This
    /// constructor projects it through the speaker-fusion gate once here.
    pub fn new(separator: BsrnnSeparator, target_embedding: &[f32; 192]) -> Self {
        let win = separator.win();
        let hop = separator.stride();
        let gate = project_embedding(target_embedding);
        let repeat_block_state = separator.new_state();
        Self { stft: StreamingStft::new(win, hop), istft: StreamingIstft::new(win, hop), separator, repeat_block_state, gate }
    }

    /// Resets all streaming state (STFT history, iSTFT overlap-add buffer,
    /// per-repeat-block LSTM state) - call when starting a new stream or
    /// switching target speakers. Does NOT recompute the speaker gate;
    /// call `set_target_embedding` for that.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.istft.reset();
        self.separator.reset_state(&mut self.repeat_block_state);
    }

    /// Re-projects a new target speaker's embedding into the cached gate -
    /// call on re-enrollment, then `reset()` to clear stream state too.
    pub fn set_target_embedding(&mut self, target_embedding: &[f32; 192]) {
        self.gate = project_embedding(target_embedding);
    }

    /// `chunk`: raw mixture samples for this call, at the model's native
    /// 16kHz. Returns finalized output samples (may be empty for small
    /// chunks, especially during warm-up - see `StreamingStft`'s docs).
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Vec<f32> {
        let (spec_real, spec_imag, t) = self.stft.push(chunk);
        if t == 0 {
            return Vec::new();
        }

        let mut subband_feature = self.separator.band_split_and_subband_norm(&spec_real, &spec_imag, t);
        self.separator.apply_speaker_gate(&mut subband_feature, &self.gate, t);
        let sep_output = self.separator.run_separator(&subband_feature, &mut self.repeat_block_state, t);
        let (est_real, est_imag) = self.separator.run_band_masker(&sep_output, &spec_real, &spec_imag, t);

        self.istft.push(&est_real, &est_imag, t)
    }
}
