mod commom;

use commom::random::{Random, get_nth_seed_value};
use commom::classes::Class;
use commom::benchmarks::Benchmark;
use std::borrow::Borrow;
use std::time::Instant;
use std::{thread::{available_parallelism, JoinHandle}};
use commom::timer::show_time;


// Value from the nasa paper
const SEED: u64 = 314_159_265;

fn main() {
    // Values for the S class
    let n:usize = 1400;
    let iterations = 15;
    let lambda = 10;
    let non_zeros = 7;


    // Create the matix and the vectors
    let mut x:Vec<f64> = Vec::new();
    
    // Main loop
    for i in 0..iterations {

        
    }


}

fn create_matrix(order:usize, non_zeros:usize){
    todo!()
}

fn conjugate_gradient(A:&mut Vec<Vec<f64>>) -> f64 {
    todo!()
}

// Return the multiplication of a * (b*scalar)
fn multiply_vectors(a: &Vec<f64>, b:&Vec<f64>, scalar: f64) -> Vec<f64>{
    todo!()
}

// Compute a = a + b * scalar
// Where a, b are two vectors with the same length and scalar is a float value that is multiplied by every
// value of b
fn multiply_add_self(a: &mut Vec<f64>, b:&Vec<f64>, scalar:f64 ) {
    todo!()
}

// Calculate the magnitude of 
fn magnitude(a:&Vec<f64>) -> f64 {
    todo!()
}