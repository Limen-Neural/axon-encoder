const MINING_DOPAMINE_DECAY: f32 = 0.97;
const EVENT_DOPAMINE_DECAY: f32 = 0.95;
const CORTISOL_DECAY: f32 = 0.90;
const ACETYLCHOLINE_DECAY: f32 = 0.99;

#[derive(Debug, Clone, Copy, Default)]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
    pub fpga_stress: f32,
    pub market_volatility: f32,
    pub mining_dopamine: f32,
}

impl NeuroModulators {
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * EVENT_DOPAMINE_DECAY).max(0.0);
        self.cortisol = (self.cortisol * CORTISOL_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.mining_dopamine *= MINING_DOPAMINE_DECAY;
    }
}
