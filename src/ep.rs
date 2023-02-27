mod commom;

use std::borrow::Borrow;

use commom::random::Random;
use commom::timer::show_time;

// This value comes from the NASA paper https://www.nas.nasa.gov/assets/pdf/techreports/1994/rnr-94-007.pdf
const SEED: u64 = 271_828_813;

fn main() {
    let mut random_source = Random::new(SEED);

    let passes: u64 = 1_000_000_000;

    let antes = std::time::Instant::now();
    let mut sum = 0.0 ;
    for _ in 0..passes {
        let a = random_source.next_f64();
        sum += a;
        // println!("{:.20}", a);
    }
    let agora = std::time::Instant::now();
    println!("sum {sum}");
    show_time((agora - antes).borrow());
}
