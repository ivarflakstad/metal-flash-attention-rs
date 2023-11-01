use std::ffi::c_void;
use std::time::Instant;

use metal::Device;

use metal_flash_attention::attention::encode_attention;
use metal_flash_attention::datatype::Float;
use metal_flash_attention::gemm::encode_gemm;
use metal_flash_attention::tensor::Tensor;

const M: usize = 4;
const N: usize = 4;
const K: usize = 4;

fn main() {
    // gemm();
    attention();
}

fn gemm() {
    let device = Device::system_default().expect("No device found");

    // let _shape_a = Shape::from([M, K]);
    // let _shape_b = Shape::from([K, N]);
    // let _shape_c = Shape::from([M, N]);

    let a = Tensor::<Float>::random(&device, vec![M, K], 5.0..7.0);
    let b = Tensor::random(&device, vec![K, N], 6.0..9.0);
    let mut c = Tensor::new(&device, vec![M, N]);
    let d = None; //Tensor::<Float>::new(&device, vec![2, 3]);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encode_gemm(
        &device, encoder, &a, &b, &mut c, &d, false, false, false, 1.0, 0.0, false,
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

fn attention() {
    let device = Device::system_default().expect("No device found");

    const B: usize = 1;
    const R: usize = 8;
    const C: usize = 8;
    const H: usize = 1;
    const D: usize = 8;

    let expected_q = Tensor::<Float>::random(&device, vec![B, R, H, D], 0.0..1.0);
    let expected_k = Tensor::random(&device, vec![B, C, H, D], 0.0..1.0);
    let expected_v = Tensor::random(&device, vec![B, C, H, D], 0.0..1.0);
    let expected_o = Tensor::new(&device, vec![B, R, H, D]);
    let expected_mask = Tensor::<Float>::random(&device, vec![1, 1, R, C], 0.0..1.0);

    let actual_q = Tensor::copy(&expected_q);
    let actual_k = Tensor::copy(&expected_k);
    let actual_v = Tensor::copy(&expected_v);
    let actual_o = Tensor::copy(&expected_o);
    let actual_mask = Tensor::copy(&expected_mask);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encode_attention(
        &device,
        encoder,
        &expected_o,
        &expected_q,
        &expected_k,
        &expected_v,
        Some(&expected_mask),
        false,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    encoder.end_encoding();
    command_buffer.commit();
    let start = Instant::now();
    command_buffer.wait_until_completed();
    let end = Instant::now();
    println!("Elapsed: {:?}", end - start);

    let num_elements = expected_o.count();
    let expected_o_contents = expected_o.contents();
    let actual_o_contents = actual_o.contents();
    for i in 0..num_elements {
        assert_eq!(expected_o_contents[i], actual_o_contents[i]);
    }
}

fn euclidean_distance<T>(a: Vec<T>, b: Vec<T>) -> f64
where
    T: Into<f64> + Clone + Copy,
{
    assert_eq!(a.len(), b.len(), "Lengths not equal");

    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += (a[i].into() - b[i].into()).powi(2);
    }

    sum.sqrt()
}

fn approx_eq<T>(
    a: Vec<T>,
    b: Vec<T>,
    avg_magnitude: f64,
    avg_deviation: f64,
    batch_size: Option<usize>,
) where
    T: Into<f64> + Clone + Copy,
{
    assert_eq!(a.len(), b.len(), "Lengths not equal");

    let tolerance = avg_magnitude.max(avg_deviation * 3e-7);

    let distance = euclidean_distance(a, b);
    assert!(
        distance < tolerance,
        "Distance not less than tolerance: {} < {} ",
        distance,
        tolerance
    );
}

pub fn read_to_vec<T: Clone>(ptr: *mut c_void, len: usize) -> Vec<T> {
    let contents_ptr = ptr as *const T;
    assert!(!contents_ptr.is_null());
    let sl = unsafe { std::slice::from_raw_parts(contents_ptr, len) };
    sl.to_vec()
}
