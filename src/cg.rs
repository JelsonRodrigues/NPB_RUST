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
    let n:usize = 2;
    let iterations = 2;
    let lambda = 10.0;
    let non_zeros = 7;


    // Create the matix and the vectors
    let mut x:Vec<f64> = vec![1.0; n];
    let mut z:Vec<f64> = vec![0.0; n];
    let mut A: Vec<Vec<f64>> = create_matrix(n, non_zeros);
    
    A[0][0] = 2.0;
    A[0][1] = 3.0;
    A[1][0] = 1.0;
    A[1][1] = 4.0;

    // Main loop
    for i in 0..iterations {
        // solve Az = x 
        let r = conjugate_gradient(&A, &mut z, &x);

        // zeta = lambda - 1 / (x * z)
        let zeta = lambda - 1.0 / multiply_vector_by_column(&x, &z, 1.0);

        // Print it, zeta, r
        println!("Iteration = {i}, ||r|| = {r}, zeta = {zeta}");

        // x = z / ||z||
        // Means that x will be a unit vector of z
        multiply_add_self(&mut x, &z, 1.0 /  magnitude(&z));
    }
}

fn create_matrix(order:usize, non_zeros:usize) -> Vec<Vec<f64>> {
    let mut matrix :Vec<Vec<f64>> = Vec::with_capacity(order);
    
    for _ in 0..order {
        matrix.push(vec![0.0; order]);
    }

    return matrix;
}

const CONJUGATE_GRADIENT_ITERATIONS :u32 = 25;

fn conjugate_gradient(A:&Vec<Vec<f64>>, z:&mut Vec<f64>, x:&Vec<f64>) -> f64 {
    // Assert A, z and x are the same size
    assert!(A.len() == z.len());
    assert!(A.len() == x.len());
    
    /*
    Change this section from alocating inside the function to receive the buffers
    The allocation and releasing of these buffers can get quite expensive, because 
    the size is huge 1400 <= x <= 9000000
    */

    // Initialize z to 0
    z.fill(0.0);

    // Initialize r to x
    // let mut r:Vec<f64> = Vec::with_capacity(x.len());
    // r = x.clone();
    let mut r = x.to_vec();

    let mut rho = multiply_vector_by_column(&r, &r, 1.0);

    // let mut p:Vec<f64> = Vec::with_capacity(r.len());
    // // p.clone_from(&r);
    // p = r.clone();

    let mut p  = r.to_vec();


    // Prealocate for q
    let mut q : Vec<f64> = vec![0.0; p.len()];

    for _ in 1..CONJUGATE_GRADIENT_ITERATIONS {
        // q = Ap
        multiply_matrix_by_vector(&A, &p, &mut q);
        // alpha = rho / (p * q)
        let alpha = rho / multiply_vector_by_column(&p, &q, 1.0);
        // z = z + alpha * p
        multiply_add_self(z, &q, alpha);
        // rho_0 = rho
        let rho_0 = rho;
        // r = r - alpha * q 
        multiply_add_self(&mut r, &q, -alpha);
        // rho = r * r
        rho = multiply_vector_by_column(&r, &r, 1.0);
        // beta = rho / rho_0
        let beta = rho / rho_0;
        // p = r + beta * p
        add_multiply_self(&mut p, &r, beta);
    }


    // The operation I'm performing here is r = || x - Az ||
    // Here i'm using the r vector as a temp to the result of Az
    // r = A * z
    multiply_matrix_by_vector(&A, &z, &mut r);
    // r = x - r
    add_multiply_self(&mut r, x, -1.0);

    return magnitude(&r);
}

// Return the multiplication of a * (b*scalar)
fn multiply_vector_by_column(a: &Vec<f64>, b:&Vec<f64>, scalar: f64) -> f64{
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut result = 0.0;
    for i in 0..len {
        result += a[i] * (b[i] * scalar);
    }

    return  result;
}

// Calculate result = matrix * vector
fn multiply_matrix_by_vector(matrix:&Vec<Vec<f64>>, vector:&Vec<f64>, result:&mut Vec<f64>) {
    assert_eq!(matrix[0].len(), vector.len());
    assert_eq!(matrix.len(), result.len());

    let vec_size = result.len();
    for i in 0..vec_size {
        result[i] = multiply_vector_by_column(&matrix[i], vector, 1.0);
    }
}

// Compute vector_a = vector_a + vector_b * scalar
fn multiply_add_self(vector_a: &mut Vec<f64>, vector_b:&Vec<f64>, scalar:f64) {
    assert_eq!(vector_a.len(), vector_b.len());

    let len = vector_a.len();
    for i in 0..len {
        vector_a[i] = vector_a[i] + vector_b[i] * scalar;
    }
}

// Compute vector_a = vector_a * scalar + vector_b
fn add_multiply_self(vector_a: &mut Vec<f64>, vector_b:&Vec<f64>, scalar: f64) {
    assert_eq!(vector_a.len(), vector_b.len());

    let len = vector_a.len();
    for i in 0..len {
        vector_a[i] = vector_a[i] * scalar + vector_b[i];
    }
}

// Calculate the magnitude of vector
fn magnitude(vector:&Vec<f64>) -> f64 {
    multiply_vector_by_column(vector, vector, 1.0).sqrt()
}