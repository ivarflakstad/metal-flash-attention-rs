use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use metal::{DeviceRef, Library, NSUInteger};
use once_cell::sync::Lazy;
use std::io;
use std::io::Write;

use crate::pipeline::{Parameters, Pipeline};
use crate::Result;

const LIB_DATA: &[u8] = include_bytes!("libMetalFlashAttention.metallib");
static METAL_FLASH_ATTENTION_LIB: Mutex<Option<Library>> = Mutex::new(None);
static PIPELINE_CACHE: Lazy<Mutex<HashMap<Parameters, Pipeline>>> =
    Lazy::new(|| Mutex::new(HashMap::default()));

pub fn get_cached_pipeline<T: Into<Parameters>>(p: T) -> Option<Pipeline> {
    if let Ok(cache) = PIPELINE_CACHE.lock() {
        cache.get(&p.into()).cloned()
    } else {
        None
    }
}

pub fn cache_pipeline<T: Into<Parameters>>(p: T, pipeline: Pipeline) -> Result<()> {
    if let Ok(mut cache) = PIPELINE_CACHE.lock() {
        cache.insert(p.into(), pipeline);
        return Ok(());
    }
    Err("Failed to cache pipeline".to_string())
}

pub fn load_mfa_lib(device: &DeviceRef) -> Result<Library> {
    if let Some(mfa_lib) = get_mfa_lib() {
        Ok(mfa_lib)
    } else {
        init_mfa_lib(device)
    }
}

fn get_mfa_lib() -> Option<Library> {
    if let Ok(lib) = METAL_FLASH_ATTENTION_LIB.lock() {
        lib.clone()
    } else {
        None
    }
}

fn init_mfa_lib(device: &DeviceRef) -> Result<Library> {
    match device.new_library_with_data(LIB_DATA) {
        Ok(lib) => {
            if let Ok(mut mfa_lib) = METAL_FLASH_ATTENTION_LIB.lock() {
                *mfa_lib = Some(lib.clone());
            }
            Ok(lib)
        }
        Err(e) => Err(format!("Failed to load MetalFlashAttention library: {}", e)),
    }
}

pub fn void_ptr<T>(v: &T) -> *const c_void {
    (v as *const T).cast()
}

pub(crate) fn ceil_divide(target: NSUInteger, granularity: u16) -> Result<NSUInteger> {
    crate::assert_result!(granularity > 0, "Granularity must be greater than 0");
    Ok((target + granularity as NSUInteger - 1) / granularity as NSUInteger)
}

#[macro_export]
macro_rules! assert_result {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err(format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! assert_eq_result {
    ($a:expr, $b:expr, $($arg:tt)*) => {
        $crate::assert_result!($a == $b, $($arg)*)
    };
}

pub fn progress_bar(i: usize, len: usize) {
    print!("\r");
    print!("[");
    print!("{}", "=".repeat(i));
    print!("{}", " ".repeat(len - i - 1));
    print!("]");
    io::stdout().flush().unwrap();
}
