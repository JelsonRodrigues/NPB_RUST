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

    /*
    This function return the seed value that will be state, after generating 
    nth random values from the current seed
     */
    pub fn get_nth_seed_value(&self, nth_position:u64) -> u64 {
        get_nth_seed_value(self.state, nth_position)
    }
}

/*
    This function return the seed value that will be state, after generating 
    nth random values from the specified seed
*/
pub fn get_nth_seed_value(starting_seed : u64, nth_position:u64) -> u64 {
    // This algorithm is based on the original paper
    let m = f64::log2(nth_position as f64) as u64 + 1;
    let mut k = nth_position;
    let mut b = starting_seed;
    let mut t = MULTIPLIER;
    for _ in 0..m {
        if k % 2 != 0 { // k is odd
            b = b.wrapping_mul(t) % MODULUS;
        }
        k = k/2;
        t = t.wrapping_pow(2) % MODULUS;
    }

    return  b;
}