// The values bellow are defined in the section 2.3 of the nasa NPB paper
// https://www.nas.nasa.gov/assets/pdf/techreports/1994/rnr-94-007.pdf

const MULTIPLIER: u64 = 1_220_703_125;      // 5^13
const MODULUS: u64 = 70_368_744_177_664;    // 2^46

#[derive(Clone, Copy)]
pub struct Random {
    state: u64,
}

#[allow(dead_code)]
impl Random {
    pub const MAX: u64 = MODULUS;

    pub fn new(seed: u64) -> Self {
        Random { state: seed }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.state = seed;
    }

    /// Generate a random float number between 0 and 1
    pub fn next_f64(&mut self) -> f64 {
        self.state = MULTIPLIER.wrapping_mul(self.state) % MODULUS;

        return self.state as f64 / Random::MAX as f64;
    }

    pub fn next_in_range_f64(&mut self, lower_bound: f64, upper_bound: f64) -> Result<f64, &str> {
        let number = self.next_f64();

        if upper_bound <= lower_bound {
            return Err("Invalid Range");
        }

        let difference = upper_bound - lower_bound;
        return Ok(number * difference + lower_bound);
    }
}
