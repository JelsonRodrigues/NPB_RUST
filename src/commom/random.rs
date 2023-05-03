// The values bellow are defined in the section 2.3 of the nasa NPB paper
// https://www.nas.nasa.gov/assets/pdf/techreports/1994/rnr-94-007.pdf

const MULTIPLIER: u64 = 1_220_703_125; // 5^13
const MODULUS: u64 = 70_368_744_177_664; // 2^46

#[derive(Clone, Copy, Debug)]
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

        self.state as f64 / Random::MAX as f64
    }

    pub fn next_in_range_f64(&mut self, lower_bound: f64, upper_bound: f64) -> Result<f64, &str> {
        let number = self.next_f64();

        if upper_bound <= lower_bound {
            return Err("Invalid Range");
        }

        let difference = upper_bound - lower_bound;
        Ok(number * difference + lower_bound)
    }

    /*
    This function return the seed value that will be state, after generating
    nth random values from the current seed
     */
    pub fn get_nth_seed_value(&self, nth_position: u64) -> u64 {
        get_nth_seed_value(self.state, nth_position)
    }
}

/*
    This function return the seed value that will be state, after generating
    nth random values from the specified seed
*/
pub fn get_nth_seed_value(starting_seed: u64, nth_position: u64) -> u64 {
    // This algorithm is based on the original paper
    let m = f64::log2(nth_position as f64) as u64 + 1;
    let mut b = starting_seed;
    let mut t = MULTIPLIER;
    for i in 0..m {
        if nth_position & (1 << i) != 0 {
            // the i-th bit is 1
            b = b.wrapping_mul(t) % MODULUS;
        }
        t = t.wrapping_pow(2) % MODULUS;
    }

    b
}

// This method is faster than the normal version, but the maximum expoent must be 32 bits
#[allow(dead_code)]
pub fn get_nth_seed_value_u32(starting_seed: u64, nth_position: u32) -> u64 {
    MULTIPLIER
        .wrapping_pow(nth_position)
        .wrapping_mul(starting_seed)
        % MODULUS
}
