pub mod random;
pub mod timer;

pub mod classes {
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy)]
    pub enum Class {
        S,
        W,
        A,
        B,
        C,
        D,
        E,
        F
    }
}

pub mod benchmarks {
    use crate::Class;
    const EPSILON: f64 = 1.0e-8;    // value estabilished in the official fotran implementation

    #[derive(Debug, Clone, Copy)]
    pub enum Benchmark {
        EP(Class),
        CG(Class),
    }

    //todo: refactor the get_difficulty functions

    impl Benchmark {
        pub fn ep_get_difficulty(&self) -> u64 {
            match self {
                Benchmark::EP(class) => {
                    match class {
                        Class::S => return 1 << 24,    // 2 ^ 24
                        Class::W => return 1 << 25,    // 2 ^ 25
                        Class::A => return 1 << 28,    // 2 ^ 28
                        Class::B => return 1 << 30,    // 2 ^ 30
                        Class::C => return 1 << 32,    // 2 ^ 32
                        Class::D => return 1 << 36,    // 2 ^ 36
                        Class::E => return 1 << 40,    // 2 ^ 40
                        Class::F => return 1 << 44,    // 2 ^ 44
                    }
                },
                Benchmark::CG(class) => todo!(),
            }
        }

        pub fn cg_get_difficulty(&self) -> (usize, usize, f64, usize) {
            match self {
                Benchmark::CG(class) => {
                    match class {
                        Class::S => return (1400, 15, 10.0, 7),
                        Class::W => return (7000, 15, 12.0, 8),
                        Class::A => return (14000, 15, 20.0, 11),  
                        Class::B => return (75000, 75, 60.0, 13),  
                        Class::C => return (150000, 75, 110.0, 15),  
                        Class::D => return (1500000, 100, 500.0, 21),  
                        Class::E => return (9000000, 100, 1500.0, 26),  
                        Class::F => return (54000000, 100, 5000.0, 31),
                    }
                },
                Benchmark::EP(class) => todo!(),
            }
        }
        

        pub fn ep_verify(&self, sum_x: f64, sum_y: f64, gaussian_count: u64) -> bool {
            let mut verified = true;
            let sum_x_verify_value;
            let sum_y_verify_value;
            let gaussian_count_verify_value;

            // reference values used comes from the official fortran implementation
            match self {
                Benchmark::EP(class) => {
                    match  class {
                        Class::S =>  {
                            sum_x_verify_value = -3.247834652034740e+3;
                            sum_y_verify_value = -6.958407078382297e+3;
                            gaussian_count_verify_value = 13176389;
                        },
                        Class::W =>  {
                            sum_x_verify_value = -2.863319731645753e+3;
                            sum_y_verify_value = -6.320053679109499e+3;
                            gaussian_count_verify_value = 26354769;
                        },
                        Class::A =>  {
                            sum_x_verify_value = -4.295875165629892e+3;
                            sum_y_verify_value = -1.580732573678431e+4;
                            gaussian_count_verify_value = 210832767;
                        },
                        Class::B =>  {
                            sum_x_verify_value =  4.033815542441498e+4;
                            sum_y_verify_value = -2.660669192809235e+4;
                            gaussian_count_verify_value = 843345606;
                        },
                        Class::C =>  {
                            sum_x_verify_value =  4.764367927995374e+4;
                            sum_y_verify_value = -8.084072988043731e+4;
                            gaussian_count_verify_value = 3373275903;
                        },
                        Class::D =>  {
                            sum_x_verify_value =  1.982481200946593e+5;
                            sum_y_verify_value = -1.020596636361769e+5;
                            gaussian_count_verify_value = 53972171957;
                        },
                        Class::E =>  {
                            sum_x_verify_value = -5.319717441530e+05;
                            sum_y_verify_value = -3.688834557731e+05;
                            gaussian_count_verify_value = 863554308186;
                        },
                        Class::F =>  {
                            sum_x_verify_value = 1.102426773788175e+13;
                            sum_y_verify_value = 1.102426773787993e+13;
                            gaussian_count_verify_value = 13816870608324;
                        },
                    }
                },
                Benchmark::CG(_) => todo!(),
                
                // _ => verified = false,
            }     

            if verified {
                let sum_x_err = ((sum_x - sum_x_verify_value) / sum_x_verify_value).abs();
                let sum_y_err = ((sum_y - sum_y_verify_value) / sum_y_verify_value).abs();
                let gaussian_count_err = ((gaussian_count - gaussian_count_verify_value) as f64 / gaussian_count_verify_value as f64).abs();
                println!("X.err: {}\nY.err: {}\nCount.err: {}", sum_x_err, sum_y_err, gaussian_count_err);
                verified = (sum_x_err <= EPSILON) && (sum_y_err <= EPSILON) && (gaussian_count_err <= EPSILON);
            }

            return verified;
        }
    }
}