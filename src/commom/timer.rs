use std::time::Duration;

pub fn show_time(duration: &Duration) {
    let ms = duration.as_millis() % 1000;
    let s = duration.as_secs() % 60;
    let m = duration.as_secs() / 60 % 60;
    let h = duration.as_secs() / 60 / 60;
    println!("\t{h}h:{m}m:{s}s:{ms}ms"); 
}