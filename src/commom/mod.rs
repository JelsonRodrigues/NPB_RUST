pub mod random;
pub mod timer;

pub mod classes {
    #[derive(Debug, Clone, Copy)]
    pub enum Class {
        S,
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
    #[derive(Debug, Clone, Copy)]
    pub enum Benchmark {
        EP(Class),
    }

    impl Benchmark {
        pub fn get_difficulty(&self) -> u64 {
            match self {
                Benchmark::EP(class) => {
                    match class {
                        Class::S => return 1 << 24,    // 2 ^ 24
                        Class::A => return 1 << 28,    // 2 ^ 28
                        Class::B => return 1 << 30,    // 2 ^ 30
                        Class::C => return 1 << 32,    // 2 ^ 32
                        Class::D => return 1 << 36,    // 2 ^ 36
                        Class::E => return 1 << 40,    // 2 ^ 40
                        Class::F => return 1 << 44,    // 2 ^ 44
                    }
                },
            }
        }
    }
}