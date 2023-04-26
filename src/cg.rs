mod commom;

use commom::random::{Random, get_nth_seed_value};
use commom::classes::Class;
use commom::benchmarks::Benchmark;
use std::borrow::Borrow;
use std::ops::Index;
use std::time::Instant;
use std::vec;
use std::{thread::{available_parallelism, JoinHandle}};
use commom::timer::show_time;


// Value from the nasa paper
const SEED: u64 = 314_159_265;

fn main() {
    // Values for the S class
    let n:usize = 1400;
    let iterations = 15;
    let lambda = 10.0;
    let non_zeros = 7;

    // Create the matix and the vectors
    let mut x:Vec<f64> = vec![1.0; n];
    let mut z:Vec<f64> = vec![0.0; n];
    let mut A = makea(n, non_zeros, lambda);
    
    // Main loop
    for i in 0..iterations {
        // solve Az = x 
        let r = conjugate_gradient(&A, &mut z, &x);

        // zeta = lambda - 1 / (x * z)
        let zeta = lambda - 1.0 / multiply_vector_by_column(&x, &z, 1.0);

        // Print it, zeta, r
        println!("Iteration = {}, ||r|| = {r}, zeta = {zeta}", i+1);

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

fn conjugate_gradient(A:&SparseMatrix, z:&mut Vec<f64>, x:&Vec<f64>) -> f64 {
    // Assert A, z and x are the same size
    // assert!(A.get_rows() == z.len());
    // assert!(A.get_columns() == x.len());
    
    /*
    Change this section from alocating inside the function to receive the buffers
    The allocation and releasing of these buffers can get quite expensive, because 
    the size is huge 1400 <= x <= 9000000
    */

    // Initialize z to 0
    z.fill(0.0);

    // Initialize r to x
    let mut r = x.to_vec();

    let mut rho = multiply_vector_by_column(&r, &r, 1.0);

    let mut p  = r.to_vec();


    // Prealocate for q
    let mut q : Vec<f64> = vec![0.0; p.len()];

    for _ in 0..CONJUGATE_GRADIENT_ITERATIONS {
        // q = Ap
        A.multiply_by_vector(&p, &mut q);
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
    // multiply_matrix_by_vector(&A, &z, &mut r);
    A.multiply_by_vector(&z, &mut r);
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

// Get the geometric progression ratio, given that the N term is equal to 0.1
// and the 1st term is equal to 1.0
// Return the value x for the expresion log_x(cond) = n
// that is equal to cond ^ (1.0 / n)
fn get_ratio(n:usize, cond:f64) -> f64 {
    return cond.powf(1.0 / n as f64);
}


// This function calculates the outer product between two vector, 
// The first must be a column vector and the second a row vector
// The result is a matrix with the same number of rows as the column vector 
// and the same number of columns as the row vector
fn outer_product(column_vector:&Vec<f64>, row_vector:&Vec<f64>) -> Vec<Vec<f64>> {
    let mut result_matrix : Vec<Vec<f64>> = Vec::with_capacity(column_vector.len());

    for row in column_vector {
        let mut new_row:Vec<f64> = Vec::with_capacity(row_vector.len());
        for column in row_vector {
            new_row.push(row * column);
        }
        result_matrix.push(new_row);
    }

    return result_matrix;
}

// Generate a sparse matrix represented as CSM with
// a given condition number, main diagonal shift and total number of non zeros


fn makea(n:usize, row_non_zeros:usize, lambda:f64) -> SparseMatrix {
    let max_non_zeros:usize = n * (row_non_zeros + 1) * (row_non_zeros + 1);
   
    // Obtain the smallest power of two that is greater or equal n
    
    let power_of_two:usize =  n.next_power_of_two();

    // generate the values and its positions in the sparse matrix, line by line
    // and add them to the auxiliar matrix. This auxiliar matrix will be basically
    // an array of randomly generated sparse vectors

    let mut sparse_matrix_aux = SparseMatrix::new(n, n);
    sparse_matrix_aux.reserve_capacity(max_non_zeros);
    let mut random_index:usize = 0;
    let mut random_value:f64 = 0.0;
    
    // In the original Fortran version, it's generated a value first
    // I could just use the second value of the SEED that is 55909509111989, but 
    // to be more in line with what is stated in the paper, 
    // I'm doing this way, the same way is done in Fortran
    let mut random_generator:Random = Random::new(SEED);
    random_generator.next_f64();
    
    for row in 0..n {
        for _ in 0..row_non_zeros {
            loop {
                random_value = random_generator.next_f64();
                random_index = (random_generator.next_f64() * power_of_two as f64) as usize;
                
                // Check is inside bounds
                if random_index >= n {continue;}

                // Check was already generated
                if sparse_matrix_aux[(row, random_index)] == 0.0 {
                    break;
                }
            }
            sparse_matrix_aux.set_index_to_value(random_value, row, random_index);
        }
        // Add 1/2 to the diagonal
        sparse_matrix_aux.set_index_to_value(0.5, row, row);
    } 
    if sparse_matrix_aux.get_non_zero_count() > max_non_zeros {
        panic!("Number of non-zeros generated for the matrix exceeded the maximum defined.\n
        Non-zero count: {}, maximum non-zeros: {}", sparse_matrix_aux.get_non_zero_count(), max_non_zeros);
    }

    let mut sparse_matrix = SparseMatrix::new(n, n);

    // Here the sparse matrix will be constructed
    let mut scale = 1.0;
    let cond_number = 0.1;
    let ratio = get_ratio(n, cond_number);

    for row_number in 0..n {
        let row_as_sparse_vector = sparse_matrix_aux.get_row_as_sparse_vector(row_number);

        outer_product_sum_on_sparse_matrix(row_as_sparse_vector, row_as_sparse_vector, &mut sparse_matrix, scale);

        scale *= ratio;
    }

    // Traverse the diagonal adding 0.1 and subtracting the shift (lambda)
    for index in 0..n {
        sparse_matrix.set_index_to_value(sparse_matrix[(index, index)] + 0.1 - lambda, index, index);
    }

    return sparse_matrix;
}


fn outer_product_sum_on_sparse_matrix(column_vector : &[(f64, usize)], row_vector : &[(f64, usize)], sparse_matrix : &mut SparseMatrix, scale:f64) {
    for (row, index_row) in column_vector {
        for (col, index_col) in row_vector {
            sparse_matrix.set_index_to_value(sparse_matrix[(*index_row, *index_col)] + row * col * scale, *index_row, *index_col);
        }
    }
}


// A sparse matrix CSR compressed
struct SparseMatrix {
    // This vector will have the non-zero values of the matrix, 
    // it's saved the f64 value and the column index of that value
    values:Vec<(f64, usize)>,

    // This vector saves the pointers for each line
    rows_pointer:Vec<usize>,

    // Number of rows
    rows:usize,

    // Number of columns
    columns:usize,

    // Constant zero value
    zero_value:f64,
}

impl SparseMatrix {
    pub fn new(rows:usize, cols:usize) -> Self {
        SparseMatrix { values: Vec::new(), rows_pointer: vec![0;rows+1], rows: rows, columns: cols, zero_value: 0.0 } 
    }
    
    pub fn get_columns(&self) -> usize {self.columns}
    pub fn get_rows(&self) -> usize {self.rows}

    pub fn get_non_zero_count(&self) -> usize {
        self.values.len()
    }

    pub fn multiply_by_vector(&self, vector:&Vec<f64>, result:&mut Vec<f64>) {
        for index in 0..self.rows {
            let row_start = self.rows_pointer[index];
            let row_end = self.rows_pointer[index + 1];

            let mut sum = 0.0;
            for index_inside_values in row_start..row_end {
                let (value, col) = self.values[index_inside_values];
                sum += value * vector[col];
            }
            result[index] = sum;
        }
    }

    pub fn get_row_as_sparse_vector(&self, row : usize) -> &[(f64, usize)] {
        return &self.values[self.rows_pointer[row]..self.rows_pointer[row+1]];
    }

    pub fn multiply_row_by_vector(&self, row:usize, vector:&Vec<f64>) -> f64 {
        if self.rows <= row {
            panic!("Index of row {row} is out of bounds {}", self.rows);
        }
        if self.columns < vector.len() {
            panic!("For multiplying matrix row by a vector column, 
            the size of each row (i.e the number of columns in the matrix) must be exactly the size of the vector column!!
            Found matrix row len is = {}, but vector len is = {}
            ", self.columns, vector.len());
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row+1];
        
        let mut sum : f64 = 0.0;
        if row_end - row_start > 0 {
            for i in row_start..row_end {
                let (value, index) = self.values[i];
                sum += value * vector[index];
            }
        }

        return sum;
    }

    pub fn reserve_capacity(&mut self, size:usize){
        self.values.reserve(size);
    }

    pub fn set_index_to_value(&mut self, value:f64, row:usize, col:usize) {
        // Check if the index is already a non-zero value,
        // if it is, then change the value
        
        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }
        
        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        // Seach for the column in the row
        let mut index = 0;
        let mut found = false;
        for i in row_start..row_end {
            if self.values[i].1 == col {
                index = i;
                found = true;
            }
        }
        
        if found {
            self.values[index] = (value, col);
        }
        else {
            self.values.push((0.0, 0));
            self.values[row_end..].rotate_right(1);
            
            // Insert in last position of the row
            self.values[row_end] = (value, col);

            // Insert sorted in the row
            for index in (row_start+1..(row_end+1)).rev() {
                if self.values[index].1 < self.values[index-1].1{
                    self.values.swap(index, index - 1);
                }
                else {
                    break;
                }
            }
            
            let rows_pointer_len = self.rows_pointer.len();
            for index in (row + 1)..rows_pointer_len {
                self.rows_pointer[index] += 1;
            }
        }
    }
}

impl Index<(usize, usize)> for SparseMatrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        // Check index is in bounds
        if self.rows <= index.0 || self.columns <= index.1 {
            panic!("Index out of bounds!!!");
        }
        else {
            // check if its a nonzero value, otherwise return 0.0
            let start = self.rows_pointer[index.0];
            let end = self.rows_pointer[index.0 + 1];

            // This line lengh is not 0
            if end - start > 0 {
                for (value, col) in &self.values[start..end] {
                    if col == &index.1 {
                        return value;
                    }
                }
            }
        }
        return &self.zero_value;
    }
}
