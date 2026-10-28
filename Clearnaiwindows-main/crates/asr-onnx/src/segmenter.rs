//! Cuts a continuous 16 kHz stream into utterances with a simple adaptive-energy speech detector.
//!
//! A segment starts after 60 ms of speech (with 300 ms of look-back kept), ends after 700 ms of silence or at 15 s,
//! and while it is open a *partial* snapshot is offered every 1.5 s so the GUI can show text as you speak.

const FRAME: usize = 320; // 20 ms
const PREROLL_FRAMES: usize = 15; // 300 ms
const START_FRAMES: usize = 3; // 60 ms above threshold
const END_SILENCE_FRAMES: usize = 35; // 700 ms
const KEEP_TAIL_FRAMES: usize = 10; // 200 ms of the trailing silence stays in the segment
const MIN_SPEECH_FRAMES: usize = 20; // shorter than 400 ms of speech: ignored (clicks, coughs)
const MAX_SEGMENT: usize = 15 * 16_000;
const PARTIAL_EVERY: usize = 24_000; // 1.5 s
const MIN_THRESHOLD: f32 = 0.006;

#[derive(Debug, PartialEq)]
pub enum Event {
    /// Snapshot of the still-open segment.
    Partial(Vec<f32>),
    /// A finished segment.
    Final(Vec<f32>),
}

pub struct Segmenter {
    pending: Vec<f32>,
    preroll: std::collections::VecDeque<Vec<f32>>,
    buf: Vec<f32>,
    in_speech: bool,
    above: usize,
    silence: usize,
    speech_frames: usize,
    noise: f32,
    last_partial_len: usize,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Segmenter {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            preroll: std::collections::VecDeque::new(),
            buf: Vec::new(),
            in_speech: false,
            above: 0,
            silence: 0,
            speech_frames: 0,
            noise: 0.003,
            last_partial_len: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    pub fn push(&mut self, x16: &[f32]) -> Vec<Event> {
        let mut events = Vec::new();
        self.pending.extend_from_slice(x16);
        let mut off = 0;
        while self.pending.len() - off >= FRAME {
            let frame = self.pending[off..off + FRAME].to_vec();
            off += FRAME;
            self.frame(frame, &mut events);
        }
        self.pending.drain(..off);
        events
    }

    fn finish(&mut self, events: &mut Vec<Event>) {
        let drop_tail = self.silence.saturating_sub(KEEP_TAIL_FRAMES) * FRAME;
        let keep = self.buf.len().saturating_sub(drop_tail);
        self.buf.truncate(keep);
        if self.speech_frames >= MIN_SPEECH_FRAMES {
            events.push(Event::Final(std::mem::take(&mut self.buf)));
        }
        self.buf.clear();
        self.in_speech = false;
        self.silence = 0;
        self.above = 0;
        self.speech_frames = 0;
        self.last_partial_len = 0;
    }

    fn frame(&mut self, frame: Vec<f32>, events: &mut Vec<Event>) {
        let rms = (frame.iter().map(|v| v * v).sum::<f32>() / FRAME as f32).sqrt();
        let thr = (self.noise * 3.0).max(MIN_THRESHOLD);
        let loud = rms > thr;
        if !self.in_speech {
            if !loud {
                self.noise = 0.95 * self.noise + 0.05 * rms.clamp(0.0005, 0.05);
            }
            self.above = if loud { self.above + 1 } else { 0 };
            if self.above >= START_FRAMES {
                self.in_speech = true;
                self.buf.clear();
                for f in self.preroll.drain(..) {
                    self.buf.extend_from_slice(&f);
                }
                self.buf.extend_from_slice(&frame);
                self.speech_frames = self.above;
                self.silence = 0;
                self.last_partial_len = 0;
            } else {
                self.preroll.push_back(frame);
                if self.preroll.len() > PREROLL_FRAMES {
                    self.preroll.pop_front();
                }
            }
            return;
        }
        self.buf.extend_from_slice(&frame);
        if loud {
            self.silence = 0;
            self.speech_frames += 1;
        } else {
            self.silence += 1;
        }
        if self.silence >= END_SILENCE_FRAMES || self.buf.len() >= MAX_SEGMENT {
            self.finish(events);
        } else if self.speech_frames >= MIN_SPEECH_FRAMES && self.buf.len() >= self.last_partial_len + PARTIAL_EVERY {
            self.last_partial_len = self.buf.len();
            events.push(Event::Partial(self.buf.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        (0..(secs * 16_000.0) as usize).map(|i| amp * (i as f32 * 0.3).sin()).collect()
    }

    #[test]
    fn splits_two_utterances_and_ignores_a_click() {
        let mut s = Segmenter::new();
        let mut ev = Vec::new();
        ev.extend(s.push(&vec![0.0; 16_000])); // 1 s silence
        ev.extend(s.push(&tone(2.0, 0.2))); // utterance 1
        ev.extend(s.push(&vec![0.0; 16_000])); // 1 s gap
        ev.extend(s.push(&tone(0.1, 0.2))); // click: too short
        ev.extend(s.push(&vec![0.0; 16_000]));
        ev.extend(s.push(&tone(1.0, 0.2))); // utterance 2
        ev.extend(s.push(&vec![0.0; 16_000]));
        let finals: Vec<usize> = ev.iter().filter_map(|e| if let Event::Final(a) = e { Some(a.len()) } else { None }).collect();
        assert_eq!(finals.len(), 2, "events: {} finals {:?}", ev.len(), finals);
        // 2 s of speech + 300 ms look-back + 200 ms tail
        assert!((finals[0] as f32 / 16_000.0 - 2.5).abs() < 0.1, "{finals:?}");
        assert!(ev.iter().any(|e| matches!(e, Event::Partial(_))), "a 2 s utterance must offer a partial");
    }

    #[test]
    fn long_speech_is_cut_at_fifteen_seconds() {
        let mut s = Segmenter::new();
        let ev = s.push(&tone(16.0, 0.2));
        assert!(ev.iter().any(|e| matches!(e, Event::Final(a) if a.len() >= MAX_SEGMENT)));
    }
}
