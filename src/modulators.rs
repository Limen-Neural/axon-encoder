const EVENT_DOPAMINE_DECAY: f32 = 0.95;
const CORTISOL_DECAY: f32 = 0.90;
const ACETYLCHOLINE_DECAY: f32 = 0.99;
const TEMPO_DECAY: f32 = 0.98;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
}

impl NeuroModulators {
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * EVENT_DOPAMINE_DECAY).max(0.0);
        self.cortisol = (self.cortisol * CORTISOL_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.tempo = (self.tempo * TEMPO_DECAY).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromodulators_decay() {
        let mut nm = NeuroModulators {
            dopamine: 1.0,
            cortisol: 1.0,
            acetylcholine: 1.0,
            tempo: 1.0,
        };
        nm.decay();
        assert!(nm.dopamine < 1.0);
        assert!(nm.cortisol < 1.0);
        assert!(nm.acetylcholine < 1.0);
        assert!(nm.tempo < 1.0);
    }
}
