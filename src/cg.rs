mod commom;

use commom::benchmarks::Benchmark;
use commom::classes::Class;
use commom::random::Random;
use commom::timer::show_time;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ops::Index;
use std::time::Instant;
use std::vec;

use rayon::prelude::*;

// Value from the nasa paper
const SEED: u64 = 314_159_265;

fn main() {
    // Setup benchmark values
    // todo!("Read class from arguments");
    let class = Class::A;
    let benchmark = Benchmark::CG(class);
    let benchmark_params = benchmark.cg_get_difficulty();

    let n: usize = benchmark_params.0;
    let iterations = benchmark_params.1;
    let lambda = benchmark_params.2;
    let non_zeros = benchmark_params.3;

    // Create the matrix and the vectors
    let mut x: Vec<f64> = vec![1.0; n];
    let mut z: Vec<f64> = vec![0.0; n];

    println!("Creating sparse matrix...");
    let before = Instant::now();
    let A = makea(n, non_zeros, lambda);
    let now = Instant::now();
    show_time((now - before).borrow());

    let sum = A
        .values
        .iter()
        .fold(0.0, |acumulated, item| acumulated + item.0);
    println!("Sum of nonzeros {sum}");

    // Prealocate here the vectors that are used inside conjugate gradient function to save allocation time
    let mut r: Vec<f64> = vec![0.0; n];
    let mut p: Vec<f64> = vec![0.0; n];
    let mut q: Vec<f64> = vec![0.0; n];

    // Start timing
    let before = Instant::now();
    let zeta = outer_loop_parallel(
        &A, &mut x, &mut z, &mut r, &mut p, &mut q, iterations, lambda,
    );
    let now = Instant::now();
    show_time((now - before).borrow());

    // Verification part
    if benchmark.cg_verify(zeta) {
        println!("Verification SUCESSFULL");
    } else {
        println!("Verification FAILED");
    }

    println!("zeta {zeta}");
}

fn outer_loop_serial(
    A: &SparseMatrix,
    x: &mut Vec<f64>,
    z: &mut Vec<f64>,
    r: &mut Vec<f64>,
    p: &mut Vec<f64>,
    q: &mut Vec<f64>,
    iterations: usize,
    lambda: f64,
) -> f64 {
    let mut zeta = 0.0;
    // Main loop
    for i in 0..iterations {
        // solve Az = x
        let r = conjugate_gradient(A, z, x, r, p, q);

        // zeta = lambda + 1 / (x * z)
        zeta = lambda + 1.0 / multiply_vector_by_column(x, z, 1.0);

        // Print it, zeta, r
        println!("Iteration = {:02}, ||r|| = {r:e}, zeta = {zeta:e}", i + 1);

        // x = z / ||z||
        // Means that x will be a unit vector of z
        replace_self(x, z, 1.0 / magnitude(z));
    }
    zeta
}

fn outer_loop_parallel(
    A: &SparseMatrix,
    x: &mut Vec<f64>,
    z: &mut Vec<f64>,
    r: &mut Vec<f64>,
    p: &mut Vec<f64>,
    q: &mut Vec<f64>,
    iterations: usize,
    lambda: f64,
) -> f64 {
    let mut zeta = 0.0;

    // Main loop
    for i in 0..iterations {
        // solve Az = x
        let r = conjugate_gradient_parallel(A, z, x, r, p, q);

        // (x * z, z * z)
        let (x_inner_product_z, z_inner_product_z) = x
            .par_iter()
            .zip_eq(z.par_iter())
            .map(|(x, z)| (x * z, z * z))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        // zeta = lambda + 1 / (x * z)
        zeta = lambda + 1.0 / x_inner_product_z;

        // Print it, zeta, r
        println!("Iteration = {:02}, ||r|| = {r:e}, zeta = {zeta:e}", i + 1);

        // x = z / ||z||
        let magnitude_z = z_inner_product_z.sqrt();
        x.par_iter_mut().zip_eq(z.par_iter()).for_each(|(x, z)| {
            *x = z / magnitude_z;
        });
    }
    zeta
}

const CONJUGATE_GRADIENT_ITERATIONS: u32 = 25;

fn conjugate_gradient(
    A: &SparseMatrix,
    z: &mut Vec<f64>,
    x: &Vec<f64>,
    r: &mut Vec<f64>,
    p: &mut Vec<f64>,
    q: &mut Vec<f64>,
) -> f64 {
    // Set z to 0.0
    z.fill(0.0);

    // Set r and p vectors to x
    *r = x.to_vec();
    *p = x.to_vec();

    // Calculate rho r * r
    let mut rho = multiply_vector_by_column(r, r, 1.0);

    for _ in 0..CONJUGATE_GRADIENT_ITERATIONS {
        // q = Ap
        A.multiply_by_vector(p, q);

        // alpha = rho / (p * q)
        let alpha = rho / multiply_vector_by_column(p, q, 1.0);

        // z = z + alpha * p
        multiply_add_self(z, p, alpha);

        // rho_0 = rho
        let rho_0 = rho;

        // r = r - alpha * q
        multiply_add_self(r, q, -alpha);

        // rho = r * r
        rho = multiply_vector_by_column(r, r, 1.0);

        // beta = rho / rho_0
        let beta = rho / rho_0;

        // p = r + beta * p
        add_multiply_self(p, r, beta);
    }

    // The operation I'm performing here is r = || x - Az ||
    // Here i'm using the r vector as a temp to the result of Az
    // r = A * z
    A.multiply_by_vector(z, r);

    // r = x - r
    add_multiply_self(r, x, -1.0);

    // return the magnitude
    magnitude(r)
}

fn conjugate_gradient_parallel(
    A: &SparseMatrix,
    z: &mut Vec<f64>,
    x: &Vec<f64>,
    r: &mut Vec<f64>,
    p: &mut Vec<f64>,
    q: &mut Vec<f64>,
) -> f64 {
    /* Parallel Section */

    let x_iter = x.par_iter();
    let r_iter = r.par_iter_mut();
    let z_iter = z.par_iter_mut();
    let p_iter = p.par_iter_mut();

    let mut rho = x_iter
        .zip_eq(r_iter)
        .zip_eq(z_iter)
        .zip_eq(p_iter)
        .map(|(((x, r), z), p)| {
            *z = 0.0;
            *r = *x;
            *p = *x;
            x * x
        })
        .reduce(|| 0.0, |a, b| a + b);

    for _ in 0..CONJUGATE_GRADIENT_ITERATIONS {
        let p_inner_product_q = A
            .rows_pointer
            .par_windows(2)
            .zip_eq(q.par_iter_mut().enumerate())
            .map(|(vec_index, (index, q))| {
                let start_row = vec_index[0];
                let end_row = vec_index[1];

                let mut sum = 0.0;
                for (value, col) in &A.values[start_row..end_row] {
                    sum += value * p[*col];
                }

                *q = sum;
                *q * p[index]
            })
            .reduce(|| 0.0, |a, b| a + b);

        let alpha = rho / p_inner_product_q;

        let rho_0 = rho;
        let z_iter = z.par_iter_mut();
        let r_iter = r.par_iter_mut();
        let p_iter = p.par_iter_mut();
        let q_iter = q.par_iter_mut();

        rho = z_iter
            .zip_eq(r_iter)
            .zip_eq(p_iter)
            .zip_eq(q_iter)
            .map(|(((z, r), p), q)| {
                *z += alpha * (*p);
                *r += -alpha * (*q);
                *r * *r
            })
            .reduce(|| 0.0, |a, b| a + b);

        // beta = rho / rho_0
        let beta = rho / rho_0;

        let r_iter = r.par_iter();
        let p_iter = p.par_iter_mut();

        r_iter.zip_eq(p_iter).for_each(|(r, p)| {
            *p = *r + beta * *p;
        });
    }

    let mag = A
        .rows_pointer
        .par_windows(2)
        .enumerate()
        .map(|(index, indexes_row)| {
            let start_row = indexes_row[0];
            let end_row = indexes_row[1];

            let mut sum = 0.0;
            for (value, col) in &A.values[start_row..end_row] {
                sum += value * z[*col];
            }

            (x[index] - sum).powi(2)
        })
        .reduce(|| 0.0, |a, b| a + b);

    mag
}

// Return the multiplication of a * (b*scalar)
fn multiply_vector_by_column(a: &Vec<f64>, b: &Vec<f64>, scalar: f64) -> f64 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut result = 0.0;
    for i in 0..len {
        result += a[i] * (b[i] * scalar);
    }

    result
}

// Compute vector_a = vector_a + vector_b * scalar
fn multiply_add_self(vector_a: &mut Vec<f64>, vector_b: &Vec<f64>, scalar: f64) {
    assert_eq!(vector_a.len(), vector_b.len());

    let len = vector_a.len();
    for i in 0..len {
        vector_a[i] += vector_b[i] * scalar;
    }
}

fn replace_self(vector_a: &mut Vec<f64>, vector_b: &Vec<f64>, scale: f64) {
    for (index, item) in vector_a.iter_mut().enumerate().take(vector_b.len()) {
        *item = vector_b[index] * scale;
    }
}

// Compute vector_a = vector_a * scalar + vector_b
fn add_multiply_self(vector_a: &mut Vec<f64>, vector_b: &Vec<f64>, scalar: f64) {
    assert_eq!(vector_a.len(), vector_b.len());

    let len = vector_a.len();
    for i in 0..len {
        vector_a[i] = vector_a[i] * scalar + vector_b[i];
    }
}

// Calculate the magnitude of vector
fn magnitude(vector: &Vec<f64>) -> f64 {
    multiply_vector_by_column(vector, vector, 1.0).sqrt()
}

// Get the geometric progression ratio, given that the N term is equal to 0.1
// and the 1st term is equal to 1.0
// Return the value x for the expresion log_x(cond) = n
// that is equal to cond ^ (1.0 / n)
fn get_ratio(n: usize, cond: f64) -> f64 {
    cond.powf(1.0 / n as f64)
}

// Generate a sparse matrix represented as CSM with
// a given condition number, main diagonal shift and total number of non zeros
fn makea(n: usize, row_non_zeros: usize, lambda: f64) -> SparseMatrix {
    let max_non_zeros_final_matrix: usize = n * (row_non_zeros + 1) * (row_non_zeros + 1);
    let max_non_zeros_sparse_vectors: usize = n * (row_non_zeros + 1);

    // Obtain the smallest power of two that is greater or equal n
    let power_of_two: usize = n.next_power_of_two();

    // generate the values and its positions in the sparse matrix, line by line
    // and add them to the auxiliar matrix. This auxiliar matrix will be basically
    // an array of randomly generated sparse vectors

    let mut sparse_matrix_aux = SparseMatrix::new(n, n);
    sparse_matrix_aux.reserve_capacity(max_non_zeros_sparse_vectors);
    let mut random_index: usize;
    let mut random_value: f64;

    // In the original Fortran version, it's generated a value first
    // I could just use the second value of the SEED that is 55909509111989, but
    // to be more in line with what is stated in the paper,
    // I'm doing this way, the same way is done in Fortran
    let mut random_generator: Random = Random::new(SEED);
    random_generator.next_f64();

    for row in 0..n {
        for _ in 0..row_non_zeros {
            loop {
                random_value = random_generator.next_f64();
                random_index = (random_generator.next_f64() * power_of_two as f64) as usize;

                // Check is inside bounds
                if random_index >= n {
                    continue;
                }

                // Check was already generated
                if sparse_matrix_aux[(row, random_index)] == 0.0 {
                    break;
                }
            }
            sparse_matrix_aux.set_index_to_value_without_updating_rows_pointer(random_value, row, random_index);
        }
        // Add 1/2 to the diagonal
        sparse_matrix_aux.set_index_to_value_without_updating_rows_pointer(0.5, row, row);
    }

    // Here the sparse matrix will be constructed
    let mut sparse_matrix = SparseMatrix::new(n, n);
    sparse_matrix.reserve_capacity(max_non_zeros_final_matrix);
    let mut scale = 1.0;
    let cond_number = 0.1;
    let ratio = get_ratio(n, cond_number);

    sparse_matrix.initialize_to_zero();
    update_row_indexes(&mut sparse_matrix, &sparse_matrix_aux);

    for row_number in 0..n {
        let row_as_sparse_vector = sparse_matrix_aux.get_row_as_sparse_vector(row_number);

        sparse_matrix.outer_product_sum_on_sparse_matrix(
            row_as_sparse_vector,
            row_as_sparse_vector,
            scale,
        );

        scale *= ratio;
    }

    // Traverse the diagonal adding 0.1 and subtracting the shift (lambda)
    for index in 0..n {
        // sparse_matrix.set_index_to_value(sparse_matrix[(index, index)] + 0.1 - lambda, index, index);
        sparse_matrix.add_index_with_value(0.1 - lambda, index, index);
    }

    sparse_matrix
}

/*
This is an optimization function, before starting the creation of the final sparse matrix
it will count the number of nonzeros that will be in each row based on the sparse vectors in the aux_matrix.
That optimization allow to preallocate the exact needed space and use the add_index_with_value function
By knowing beforehand the size of each row, its possible to make changes to only that slice in the final matrix.
 */
fn update_row_indexes(matrix: &mut SparseMatrix, aux_matrix: &SparseMatrix) {
    matrix.rows_pointer.fill(0);
    for i in 0..aux_matrix.rows {
        let start_row = aux_matrix.rows_pointer[i];
        let end_row = aux_matrix.rows_pointer[i + 1];
        let row_lenght = end_row - start_row;
        for j in start_row..end_row {
            matrix.rows_pointer[aux_matrix.values[j].1] += row_lenght;
        }
    }
    let mut sum = 0;
    for i in 0..matrix.rows_pointer.len() {
        let copy = matrix.rows_pointer[i];
        matrix.rows_pointer[i] = sum;
        sum += copy;
    }
}

// A sparse matrix CSR compressed
struct SparseMatrix {
    // This vector will have the non-zero values of the matrix,
    // it's saved the f64 value and the column index of that value
    values: Vec<(f64, usize)>,

    // This vector saves the pointers for each line
    rows_pointer: Vec<usize>,

    // Number of rows
    rows: usize,

    // Number of columns
    columns: usize,

    // Constant zero value
    zero_value: f64,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            values: Vec::new(),
            rows_pointer: vec![0; rows + 1],
            rows,
            columns: cols,
            zero_value: 0.0,
        }
    }

    pub fn get_columns(&self) -> usize {
        self.columns
    }
    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_non_zero_count(&self) -> usize {
        self.values.len()
    }

    pub fn multiply_by_vector(&self, vector: &Vec<f64>, result: &mut Vec<f64>) {
        for (index, item) in result.iter_mut().enumerate().take(self.rows) {
            let row_start = self.rows_pointer[index];
            let row_end = self.rows_pointer[index + 1];

            let mut sum = 0.0;
            for index_inside_values in row_start..row_end {
                let (value, col) = self.values[index_inside_values];
                sum += value * vector[col];
            }
            *item = sum;
        }
    }

    pub fn get_row_as_sparse_vector(&self, row: usize) -> &[(f64, usize)] {
        &self.values[self.rows_pointer[row]..self.rows_pointer[row + 1]]
    }

    pub fn multiply_row_by_vector(&self, row: usize, vector: &Vec<f64>) -> f64 {
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
        let row_end = self.rows_pointer[row + 1];

        let mut sum: f64 = 0.0;
        if row_end - row_start > 0 {
            for i in row_start..row_end {
                let (value, index) = self.values[i];
                sum += value * vector[index];
            }
        }

        sum
    }

    pub fn reserve_capacity(&mut self, size: usize) {
        self.values.reserve(size);
    }

    /* Otimization functions */

    pub fn initialize_to_zero(&mut self) {
        self.values = vec![(0.0, 0); self.values.capacity()];
        self.rows_pointer = vec![0; self.rows_pointer.capacity()];
    }


    /*
    This function does not update the rows_pointer after the insertion, it's very
    useful if you already know the number of nonzeros in each row.
    The rows pointer should all be initialized with the correct value of the size of each row
    and the vector holding the non zeros should already be fully alocated, and initialized with 0.0
     */
    pub fn add_index_with_value(&mut self, value: f64, row: usize, col: usize) {
        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search = &self.values[row_start..row_end].binary_search_by(|tuple_value| {
            if tuple_value.0 == 0.0 {
                Ordering::Greater
            } else {
                tuple_value.1.cmp(&col)
            }
        });

        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found].0 += value;
            }
            Err(index_to_insert) => {
                if self.values[row_start + index_to_insert].0 != 0.0
                    && row_start + index_to_insert <= row_end
                {
                    self.values[row_start + index_to_insert..row_end].rotate_right(1);
                }
                self.values[row_start + index_to_insert] = (value, col);
            }
        }
    }

    fn outer_product_sum_on_sparse_matrix(
        &mut self,
        column_vector: &[(f64, usize)],
        row_vector: &[(f64, usize)],
        scale: f64,
    ) {
        for (row_value, index_row) in column_vector {
            for (col_value, index_col) in row_vector {
                self.add_index_with_value(row_value * col_value * scale, *index_row, *index_col);
            }
        }
    }
    /*
    This function doesn't update the rows index, it is useful for insertion in only the last row.
    After going to the next row, it should not be inserted in a row before with this method.
     */

    pub fn set_index_to_value_without_updating_rows_pointer(&mut self, value: f64, row: usize, col: usize) {
        // Check if the index is already a non-zero value,
        // if it is, then change the value

        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }

        if self.rows_pointer[row] >= self.rows_pointer[row+1] {
            self.rows_pointer[row + 1] = self.rows_pointer[row];
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search =
            &self.values[row_start..row_end].binary_search_by(|value| value.1.cmp(&col));

        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found] = (value, col);
            }
            Err(index_to_insert) => {
                self.values
                    .insert(row_start + index_to_insert, (value, col));

                self.rows_pointer[row+1] += 1;
            }
        }
    }

    /* End of otimization functions */

    pub fn set_index_to_value(&mut self, value: f64, row: usize, col: usize) {
        // Check if the index is already a non-zero value,
        // if it is, then change the value

        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search =
            &self.values[row_start..row_end].binary_search_by(|value| value.1.cmp(&col));

        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found] = (value, col);
            }
            Err(index_to_insert) => {
                self.values
                    .insert(row_start + index_to_insert, (value, col));

                // That part is very slow, because is necessary to update the rows pointer of all the 
                // next rows
                let rows_pointer_len = self.rows_pointer.len();
                for index in (row + 1)..rows_pointer_len {
                    self.rows_pointer[index] += 1;
                }
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
        } else {
            // check if its a nonzero value, otherwise return 0.0
            let start = self.rows_pointer[index.0];
            let end = self.rows_pointer[index.0 + 1];

            // This line lengh is not 0
            if end > start {
                let result_search =
                    &self.values[start..end].binary_search_by(|value| value.1.cmp(&index.1));
                if let Ok(index_found_from_begining_of_slice) = result_search {
                    return &self.values[*index_found_from_begining_of_slice + start].0;
                }
            }
        }
        &self.zero_value
    }
}
