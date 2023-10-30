use std::ffi::c_void;
use std::time::Instant;

use metal::Device;

use metal_flash_attention::datatype::Float;
use metal_flash_attention::gemm::encode_gemm;
use metal_flash_attention::tensor::Tensor;

const M: usize = 4;
const N: usize = 4;
const K: usize = 4;

fn main() {
    let device = Device::system_default().expect("No device found");

    // let _shape_a = Shape::from([M, K]);
    // let _shape_b = Shape::from([K, N]);
    // let _shape_c = Shape::from([M, N]);

    let a = Tensor::<Float>::random(&device, vec![M, K], 5.0..7.0);
    let b = Tensor::random(&device, vec![K, N], 6.0..9.0);
    let c = Tensor::new(&device, vec![M, N]);
    let d = None; //Tensor::<Float>::new(&device, vec![2, 3]);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encode_gemm(
        &device, encoder, &a, &b, &c, &d, false, false, false, 1.0, 0.0, false,
    )
    .unwrap();
    encoder.end_encoding();
    command_buffer.commit();
    let start = Instant::now();
    command_buffer.wait_until_completed();
    let end = Instant::now();
    println!("Elapsed: {:?}", end - start);

    println!("c: {:?}", a.contents());
    println!("b: {:?}", b.contents());
    println!("c: {:?}", c.contents());
}

pub fn read_to_vec<T: Clone>(ptr: *mut c_void, len: usize) -> Vec<T> {
    let contents_ptr = ptr as *const T;
    assert!(!contents_ptr.is_null());
    let sl = unsafe { std::slice::from_raw_parts(contents_ptr, len) };
    sl.to_vec()
}
