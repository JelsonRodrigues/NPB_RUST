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
        F,
    }
}

pub mod benchmarks {
    use crate::Class;
    const EPSILON: f64 = 1.0e-8; // value estabilished in the official fotran implementation

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
                        Class::S => 1 << 24, // 2 ^ 24
                        Class::W => 1 << 25, // 2 ^ 25
                        Class::A => 1 << 28, // 2 ^ 28
                        Class::B => 1 << 30, // 2 ^ 30
                        Class::C => 1 << 32, // 2 ^ 32
                        Class::D => 1 << 36, // 2 ^ 36
                        Class::E => 1 << 40, // 2 ^ 40
                        Class::F => 1 << 44, // 2 ^ 44
                    }
                }
                Benchmark::CG(_class) => todo!(),
            }
        }

        pub fn cg_get_difficulty(&self) -> (usize, usize, f64, usize) {
            match self {
                Benchmark::CG(class) => match class {
                    Class::S => (1400, 15, 10.0, 7),
                    Class::W => (7000, 15, 12.0, 8),
                    Class::A => (14000, 15, 20.0, 11),
                    Class::B => (75000, 75, 60.0, 13),
                    Class::C => (150000, 75, 110.0, 15),
                    Class::D => (1500000, 100, 500.0, 21),
                    Class::E => (9000000, 100, 1500.0, 26),
                    Class::F => (54000000, 100, 5000.0, 31),
                },
                Benchmark::EP(_class) => todo!(),
            }
        }

        pub fn cg_verify(&self, zeta: f64) -> bool {
            let epsilon = 1.0e-10;
            if let Benchmark::CG(class) = self {
                match class {
                    Class::S => return (8.5971775078648 - zeta).abs() <= epsilon,
                    Class::W => return (10.362595087124 - zeta).abs() <= epsilon,
                    Class::A => return (17.130235054029 - zeta).abs() <= epsilon,
                    Class::B => return (22.712745482631 - zeta).abs() <= epsilon,
                    Class::C => return (28.973605592845 - zeta).abs() <= epsilon,
                    Class::D => return (52.514532105794 - zeta).abs() <= epsilon,
                    Class::E => return (77.522164599383 - zeta).abs() <= epsilon,
                    Class::F => return (107.30708264330 - zeta).abs() <= epsilon,
                }
            }

            false
        }

        pub fn ep_verify(&self, sum_x: f64, sum_y: f64, gaussian_count: u64) -> bool {
            let mut verified = true;
            let sum_x_verify_value;
            let sum_y_verify_value;
            let gaussian_count_verify_value;

            // reference values used comes from the official fortran implementation
            match self {
                Benchmark::EP(class) => match class {
                    Class::S => {
                        sum_x_verify_value = -3.247_834_652_034_74e3;
                        sum_y_verify_value = -6.958_407_078_382_297e3;
                        gaussian_count_verify_value = 13176389;
                    }
                    Class::W => {
                        sum_x_verify_value = -2.863_319_731_645_753e3;
                        sum_y_verify_value = -6.320_053_679_109_499e3;
                        gaussian_count_verify_value = 26354769;
                    }
                    Class::A => {
                        sum_x_verify_value = -4.295_875_165_629_892e3;
                        sum_y_verify_value = -1.580_732_573_678_431e4;
                        gaussian_count_verify_value = 210832767;
                    }
                    Class::B => {
                        sum_x_verify_value = 4.033_815_542_441_498e4;
                        sum_y_verify_value = -2.660_669_192_809_235e4;
                        gaussian_count_verify_value = 843345606;
                    }
                    Class::C => {
                        sum_x_verify_value = 4.764_367_927_995_374e4;
                        sum_y_verify_value = -8.084_072_988_043_731e4;
                        gaussian_count_verify_value = 3373275903;
                    }
                    Class::D => {
                        sum_x_verify_value = 1.982_481_200_946_593e5;
                        sum_y_verify_value = -1.020_596_636_361_769e5;
                        gaussian_count_verify_value = 53972171957;
                    }
                    Class::E => {
                        sum_x_verify_value = -5.319717441530e+05;
                        sum_y_verify_value = -3.688834557731e+05;
                        gaussian_count_verify_value = 863554308186;
                    }
                    Class::F => {
                        sum_x_verify_value = 1.102_426_773_788_175e13;
                        sum_y_verify_value = 1.102_426_773_787_993e13;
                        gaussian_count_verify_value = 13816870608324;
                    }
                },
                Benchmark::CG(_) => todo!(),
                // _ => verified = false,
            }

            if verified {
                let sum_x_err = ((sum_x - sum_x_verify_value) / sum_x_verify_value).abs();
                let sum_y_err = ((sum_y - sum_y_verify_value) / sum_y_verify_value).abs();
                let gaussian_count_err = ((gaussian_count - gaussian_count_verify_value) as f64
                    / gaussian_count_verify_value as f64)
                    .abs();
                println!(
                    "X.err: {}\nY.err: {}\nCount.err: {}",
                    sum_x_err, sum_y_err, gaussian_count_err
                );
                verified = (sum_x_err <= EPSILON)
                    && (sum_y_err <= EPSILON)
                    && (gaussian_count_err <= EPSILON);
            }

            verified
        }
    }
}
