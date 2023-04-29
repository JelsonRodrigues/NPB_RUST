mod commom;

use commom::random::Random;
use commom::classes::Class;
use commom::benchmarks::Benchmark;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ops::Index;
use std::time::Instant;
use std::vec;
use std::{thread::{available_parallelism, JoinHandle}};
use commom::timer::show_time;
use scoped_pool::Pool;

// Value from the nasa paper
const SEED: u64 = 314_159_265;

fn main() {
    // Setup benchmark values
    // todo!("Read class from arguments");
    let class = Class::W;
    let benchmark = Benchmark::CG(class);
    let benchmark_params = benchmark.cg_get_difficulty();

    let n:usize = benchmark_params.0;
    let iterations = benchmark_params.1;
    let lambda = benchmark_params.2;
    let non_zeros = benchmark_params.3;

    // Create the matrix and the vectors
    let mut x:Vec<f64> = vec![1.0; n];
    let mut z:Vec<f64> = vec![0.0; n];

    let antes = Instant::now();
    let A = makea(n, non_zeros, lambda);
    let agora = Instant::now();
    show_time((agora - antes).borrow());


    let sum = A.values.iter().fold(0.0, |acumulated, item| acumulated + item.0);
    println!("Sum of nonzeros {sum}");
    
    // Prealocate here the vectors that are used inside conjugate gradient function to save allocation time
    let mut r:Vec<f64> = vec![0.0; n];
    let mut p:Vec<f64> = vec![0.0; n];
    let mut q:Vec<f64> = vec![0.0; n];

    let pool = Pool::new(12);

    // Start timing
    let before = Instant::now();
    let mut zeta = 0.0;
    // Main loop
    for i in 0..iterations {
        // solve Az = x 
        // let r = conjugate_gradient(&A, &mut z, &x, &mut r, &mut p, &mut q);
        let r = conjugate_gradient_parallel(&A, &mut z, &x, &mut r, &mut p, &mut q, &pool);

        // zeta = lambda + 1 / (x * z)
        zeta = lambda + 1.0 / multiply_vector_by_column(&x, &z, 1.0);

        // Print it, zeta, r
        println!("Iteration = {:02}, ||r|| = {r:e}, zeta = {zeta:e}", i+1);

        // x = z / ||z||
        // Means that x will be a unit vector of z
        replace_self(&mut x, &z, 1.0 / magnitude(&z));
    }
    let now = Instant::now();
    show_time((now - before).borrow());

    // Verification part
    
    if  benchmark.cg_verify(zeta) {
        println!("Verification SUCESSFULL");
    }
    else {
        println!("Verification FAILED");
    }

    println!("zeta {zeta}");
}

const CONJUGATE_GRADIENT_ITERATIONS :u32 = 25;

fn conjugate_gradient(
    A:&SparseMatrix,
    z:&mut Vec<f64>, 
    x:&Vec<f64>, 
    r:&mut Vec<f64>, 
    p:&mut Vec<f64>, 
    q:&mut Vec<f64>
) -> f64 
{
    // Set z to 0.0
    z.fill(0.0);

    // Set r and p vectors to x
    *r = x.to_vec();
    *p = x.to_vec();

    // Calculate rho r * r
    let mut rho = multiply_vector_by_column(&r, &r, 1.0);

    for _ in 0..CONJUGATE_GRADIENT_ITERATIONS {
        // q = Ap
        A.multiply_by_vector(&p, q);

        // alpha = rho / (p * q)
        let alpha = rho / multiply_vector_by_column(&p, &q, 1.0);

        // z = z + alpha * p
        multiply_add_self(z, &p, alpha);

        // rho_0 = rho
        let rho_0 = rho;

        // r = r - alpha * q 
        multiply_add_self(r, &q, -alpha);

        // rho = r * r
        rho = multiply_vector_by_column(&r, &r, 1.0);

        // beta = rho / rho_0
        let beta = rho / rho_0;
        
        // p = r + beta * p
        add_multiply_self(p, &r, beta);
    }

    // The operation I'm performing here is r = || x - Az ||
    // Here i'm using the r vector as a temp to the result of Az
    // r = A * z
    A.multiply_by_vector(&z, r);

    // r = x - r
    add_multiply_self(r, x, -1.0);

    // return the magnitude
    return magnitude(&r);
}

use itertools::izip;
const CHUNK_SIZE:usize = 128;
fn conjugate_gradient_parallel(
    A:&SparseMatrix, 
    z:&mut Vec<f64>, 
    x:&Vec<f64>, 
    r:&mut Vec<f64>, 
    p:&mut Vec<f64>, 
    q:&mut Vec<f64>,
    pool: &Pool,
) -> f64  
{
    /* Parallel Section */
    let mut rho = 0.0;
    
    let x_chunks = x.chunks(CHUNK_SIZE);
    let r_chunks = r.chunks_mut(CHUNK_SIZE);
    let z_chunks = z.chunks_mut(CHUNK_SIZE);
    let p_chunks = p.chunks_mut(CHUNK_SIZE);

    pool.scoped( move |scope| {
        for (x_chunk, z_chunk, r_chunk, p_chunk) in izip!(x_chunks, z_chunks, r_chunks, p_chunks) {
            scope.execute( move || {  
                for (x, z, r, p) in izip!(x_chunk, z_chunk, r_chunk, p_chunk){
                    *z = 0.0;
                    *r = *x;
                    *p = *x;
                }
            });
        }
    });

    for i in 0..x.len() {
    //     z[i] = 0.0;
    //     r[i] = x[i];
    //     p[i] = x[i];
        rho += x[i] * x[i];
    }

    /* End Parallel Section */

    for _ in 0..CONJUGATE_GRADIENT_ITERATIONS {
        /* Paralel section */
        let mut p_inner_product_q = 0.0;
        for i in 0..A.rows {
            let start_row = A.rows_pointer[i];
            let end_row = A.rows_pointer[i+1];

            let mut sum = 0.0;
            for (value, col) in &A.values[start_row..end_row] {
                sum += value * p[*col];
            }
            q[i] = sum;
            p_inner_product_q += q[i] * p[i];
        }
        /* END Parallel Section */
        
        let alpha = rho / p_inner_product_q;

        
        let z_chunks = z.chunks_mut(CHUNK_SIZE);
        let r_chunks = r.chunks_mut(CHUNK_SIZE);
        let p_chunks = p.chunks_mut(CHUNK_SIZE);
        let q_chunks = q.chunks_mut(CHUNK_SIZE);

        pool.scoped( move |scope| {
            for (z_chunk, r_chunk, p_chunk, q_chunk) in izip!(z_chunks, r_chunks, p_chunks, q_chunks) {
                scope.execute( move || {  
                    for (z, r, p, q) in izip!(z_chunk, r_chunk, p_chunk, q_chunk){
                        *z += alpha * (*p);
                        *r += -alpha * (*q);
                    }
                });
            }
        });

        /* Paralell Section */
        let rho_0 = rho;
        rho = 0.0;
        for i in 0..x.len() {
            // z[i] += alpha * p[i];
            // r[i] += -alpha * q[i];
            rho += r[i] * r[i];
        }
        /* END Parallel Section */

        // beta = rho / rho_0
        let beta = rho / rho_0;

        let r_chunks = r.chunks(CHUNK_SIZE);
        let p_chunks = p.chunks_mut(CHUNK_SIZE);

        pool.scoped( move |scope| {
            for (r_chunk, p_chunk) in izip!(r_chunks, p_chunks) {
                scope.execute( move || {  
                    for (r, p) in izip!(r_chunk, p_chunk){
                        *p = *r + beta * *p;
                    }
                });
            }
        });

        /* Parallel Section */
        // p = r + beta * p
        // for i in 0..x.len() {
        //     p[i] = r[i] + beta * p[i];
        // }
        /* END Parallel Section */
    }


    /* Paralell Section */

    let mut mag = 0.0;
    for i in 0..A.rows {
        let start_row = A.rows_pointer[i];
        let end_row = A.rows_pointer[i+1];

        let mut sum = 0.0;
        for (value, col) in &A.values[start_row..end_row] {
            sum += value * z[*col];
        }
        mag += (x[i] - sum).powi(2);
    }

    return mag;

    /* END Parallel Section */
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

// Compute vector_a = vector_a + vector_b * scalar
fn multiply_add_self(vector_a:&mut Vec<f64>, vector_b:&Vec<f64>, scalar:f64) {
    assert_eq!(vector_a.len(), vector_b.len());

    let len = vector_a.len();
    for i in 0..len {
        vector_a[i] = vector_a[i] + vector_b[i] * scalar;
    }
}

fn replace_self(vector_a: &mut Vec<f64>, vector_b:&Vec<f64>, scale:f64) {
    for (index, item) in vector_a.iter_mut().enumerate().take(vector_b.len()){
        *item = vector_b[index] * scale;
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

// Generate a sparse matrix represented as CSM with
// a given condition number, main diagonal shift and total number of non zeros
fn makea(n:usize, row_non_zeros:usize, lambda:f64) -> SparseMatrix {
    let max_non_zeros_final_matrix:usize =  n * (row_non_zeros + 1) * (row_non_zeros + 1); 
    let max_non_zeros_sparse_vectors:usize = n * (row_non_zeros + 1);
    
    // Obtain the smallest power of two that is greater or equal n
    let power_of_two:usize = n.next_power_of_two();

    // generate the values and its positions in the sparse matrix, line by line
    // and add them to the auxiliar matrix. This auxiliar matrix will be basically
    // an array of randomly generated sparse vectors

    let mut sparse_matrix_aux = SparseMatrix::new(n, n);
    sparse_matrix_aux.reserve_capacity(max_non_zeros_sparse_vectors);
    let mut random_index:usize;
    let mut random_value:f64;
    
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
    
    // Here the sparse matrix will be constructed
    let mut sparse_matrix = SparseMatrix::new(n, n);
    sparse_matrix.reserve_capacity(max_non_zeros_final_matrix);
    let mut scale = 1.0;
    let cond_number = 0.1;
    let ratio = get_ratio(n, cond_number);
    
    // sparse_matrix.initialize_to_zero((row_non_zeros + 1) * (row_non_zeros + 1));
    for row_number in 0..n {
        let row_as_sparse_vector = sparse_matrix_aux.get_row_as_sparse_vector(row_number);

        outer_product_sum_on_sparse_matrix(row_as_sparse_vector, row_as_sparse_vector, &mut sparse_matrix, scale);
        // sparse_matrix.outer_product_sum_on_sparse_matrix(row_as_sparse_vector, row_as_sparse_vector, scale);

        scale *= ratio;
    }
    // sparse_matrix.desfragmentate();

    // Traverse the diagonal adding 0.1 and subtracting the shift (lambda)
    for index in 0..n {
        // sparse_matrix.set_index_to_value(sparse_matrix[(index, index)] + 0.1 - lambda, index, index);
        sparse_matrix.add_index_with_value_optimized(0.1 - lambda, index, index);
    }

    return sparse_matrix;
}

fn outer_product_sum_on_sparse_matrix(column_vector : &[(f64, usize)], row_vector : &[(f64, usize)], sparse_matrix : &mut SparseMatrix, scale:f64) {
    for (row_value, index_row) in column_vector {
        for (col_value, index_col) in row_vector {
            // sparse_matrix.set_index_to_value(sparse_matrix[(*index_row, *index_col)] + row * col * scale, *index_row, *index_col);
            sparse_matrix.add_index_with_value(row_value * col_value * scale, *index_row, *index_col);
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

    /* Otimization functions */
    /*
    The order to call is
    let mut a = SparseMatrix::new();
    a.reserve_capacity(value);
    a.initialize_to_zero(max_row);
    a.add_position();
    a.desfragmentate();
     */

    pub fn initialize_to_zero(&mut self, max_non_zeros_per_row:usize) {
        self.values = vec![(0.0, 0); self.values.capacity()];

        let mut value = 0;
        for i in &mut self.rows_pointer {
            *i = value;
            value += max_non_zeros_per_row;
        }
    }

    pub fn desfragmentate(&mut self) {
        let mut index_swap = 0;

        for ind_row in 0..self.rows_pointer.len()-1 {
            let row_start = self.rows_pointer[ind_row];
            let row_end = self.rows_pointer[ind_row+1];

            let new_row_start = index_swap;

            for ind in row_start..row_end {
                if self.values[ind].0 != 0.0 {
                    self.values.swap(ind, index_swap);
                    index_swap+=1;
                }
            }

            self.rows_pointer[ind_row] = new_row_start;
        }

        let last_index = self.rows_pointer.len()-1;
        self.rows_pointer[last_index] = index_swap;
    }

    pub fn add_index_with_value_optimized(&mut self, value:f64, row:usize, col:usize) {
        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search = &self.values[row_start..row_end]
            .binary_search_by(|tuple_value| {
                if tuple_value.0 == 0.0 {Ordering::Greater}
                else {tuple_value.1.cmp(&col)}
            });

        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found].0 += value;
            },
            Err(index_to_insert) => {
                if self.values[row_start + index_to_insert].0 != 0.0 && row_start + index_to_insert <= row_end  {
                    self.values[row_start + index_to_insert..row_end].rotate_right(1);
                }
                self.values[row_start + index_to_insert] = (value, col);
            },
        }
    }

    fn outer_product_sum_on_sparse_matrix(&mut self, column_vector : &[(f64, usize)], row_vector : &[(f64, usize)], scale:f64) {
        for (row_value, index_row) in column_vector {
            for (col_value, index_col) in row_vector {
                self.add_index_with_value_optimized(row_value * col_value * scale, *index_row, *index_col);
            }
        }
    }
    /* End of otimization functions */

    pub fn set_index_to_value(&mut self, value:f64, row:usize, col:usize) {
        // Check if the index is already a non-zero value,
        // if it is, then change the value
        
        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }
        
        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search = &self.values[row_start..row_end].binary_search_by(|value| value.1.cmp(&col));
        
        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found] = (value, col);
            },
            Err(index_to_insert) => {
                self.values.insert(row_start + index_to_insert, (value, col));

                /* Aqui eu nao posso atualizar tudo ate o final, utilizar alguma forma de verificacao */
                let rows_pointer_len = self.rows_pointer.len();
                for index in (row + 1)..rows_pointer_len {
                    self.rows_pointer[index] += 1;
                }
            },
        }
    }

    pub fn add_index_with_value(&mut self, value:f64, row:usize, col:usize) {
        if self.rows <= row || self.columns <= col {
            panic!("Index [{row}][{col}] out of bounds!!!");
        }

        let row_start = self.rows_pointer[row];
        let row_end = self.rows_pointer[row + 1];

        let result_search = &self.values[row_start..row_end].binary_search_by(|value| value.1.cmp(&col));

        match result_search {
            Ok(index_found) => {
                self.values[row_start + index_found].0 += value;
            },
            Err(index_to_insert) => {
                self.values.insert(row_start + index_to_insert, (value, col));

                /* Aqui eu nao posso atualizar tudo ate o final, utilizar alguma forma de verificacao */
                let rows_pointer_len = self.rows_pointer.len();
                for index in (row + 1)..rows_pointer_len {
                    self.rows_pointer[index] += 1;
                }
            },
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
                let result_search = &self.values[start..end].binary_search_by(|value| value.1.cmp(&index.1));
                if let Ok(index_found_from_begining_of_slice) = result_search {
                    return &self.values[*index_found_from_begining_of_slice + start].0;
                }
            }
        }
        return &self.zero_value;
    }
}
