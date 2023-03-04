mod commom;

use commom::random::{Random, get_nth_seed_value};
use commom::classes::Class;
use commom::benchmarks::Benchmark;
use std::borrow::Borrow;
use std::time::Instant;
use std::{thread::{available_parallelism, JoinHandle}};
use commom::timer::show_time;

// This value comes from the NASA paper https://www.nas.nasa.gov/assets/pdf/techreports/1994/rnr-94-007.pdf
const SEED: u64 = 271_828_183;

fn main() {
    // Setup benchmark values
    // todo!("Read class from arguments");
    let class = Class::C;
    let benchmark = Benchmark::EP(class);
    let n = benchmark.get_difficulty();
    
    // Value from the oficial fortran implementation
    let epsilon = 1.0e-8;
    
    // todo!("Read number of threads to use from arguments");
    let number_of_threads = available_parallelism().expect("Unable to get the total number of threads").get() as u64;
    let computations_per_thread = n / number_of_threads;
    let mut threads_handler:Vec<JoinHandle<(Vec<u64>, (f64, f64))>> = Vec::with_capacity(number_of_threads as usize);

    let before = Instant::now();
    // Creation of threads
    for index in 0..number_of_threads {
        let thread_seed_sequence = 2 * computations_per_thread * index;
        
        // If the total number cannot be divided evenly between the number of threads, ((2^28) / 6 = 44.739.242,67)
        // each thread will do the integer part of the division and the last thread will do the remaining job
        // This is an lazy way of dividing the job, because is possible to the last thread have
        // execute more ammount of work that other threads individualy
        // A future thing to do is to redivide the remaining work more evenly between the threads
        // but has to be taken in consideration the seed offset, the way is done now, that is not a problem
        let computations_per_thread = if index == number_of_threads - 1 {n - computations_per_thread * index} else {computations_per_thread};

        let thread_handler = std::thread::spawn(move || {
            calculate(get_nth_seed_value(SEED, thread_seed_sequence), computations_per_thread)
        });

        threads_handler.push(thread_handler);
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut Q = vec![0_u64;10];

    for thread in threads_handler {
        let result = thread.join().expect("Error joining the thread!!!");

        let (local_Q, (local_sum_x, local_sum_y)) = result;

        (0..Q.len()).for_each(|index| {
            Q[index] += local_Q[index];
        });

        sum_x += local_sum_x;
        sum_y += local_sum_y;
    }

    // Verification part

    println!("\nThreads {number_of_threads}");
    println!("Class {:?}", class);
    println!("Sum X {sum_x}");
    println!("Sum y {sum_y}");
    println!("Q {:?}", Q);
    println!("Sum of Q {}", Q.into_iter().sum::<u64>());

    let now = Instant::now();

    show_time((now - before).borrow());

}

// Return the sum of values in each class of the Q, a Vec<u64>
// and a tuple with the sum of Xk and Yk values
fn calculate(seed: u64, number_of_calculations:u64) -> (Vec<u64>, (f64, f64)) {
    let mut random = Random::new(seed);
    let mut Q = vec![0_u64;10];
    let mut sum_x_k = 0.0;
    let mut sum_y_k = 0.0;
    
    let transform_clojure = |value:f64| -> f64 {2.0 * value - 1.0};
    for _ in 0..number_of_calculations {
        let x = transform_clojure(random.next_f64());
        let y = transform_clojure(random.next_f64());

        // let t = x.powi(2) + y.powi(2);
        let t = x * x + y * y;

        if t <= 1.0 {
            let formula = f64::sqrt(-2.0*t.ln() / t);
            let x_k = x * formula;
            let y_k = y * formula;
            
            let highest = if x_k.abs() > y_k.abs() {x_k.abs()} else {y_k.abs()};
            let index = highest as usize;
            Q[index] += 1;

            sum_x_k += x_k;
            sum_y_k += y_k;
        }
    }

    return (Q, (sum_x_k, sum_y_k));
}