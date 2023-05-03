use std::time::Duration;

pub fn show_time(duration: &Duration) {
    let ms = duration.as_millis() % 1000;
    let s = duration.as_secs() % 60;
    let m = duration.as_secs() / 60 % 60;
    let h = duration.as_secs() / 60 / 60;
    println!("Total time: {}s", duration.as_secs_f64());
    println!("\t{h:02}h:{m:02}m:{s:02}s:{ms:03}ms");
}
