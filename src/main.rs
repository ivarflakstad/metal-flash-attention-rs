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
    gemm_performance();
    // attention();
}

fn gemm_performance() {
    const M: usize = 1536;
    const N: usize = 1536;
    const K: usize = 1536;

    type T = Float;

    const ITERATIONS: usize = 5000;

    println!("Performance: ");
    println!("{M}x{K}xf32 * {K}x{N}xf32 = {M}x{N}xf32");

    let device = Device::system_default().expect("No device found");

    let a = Tensor::<T>::random(&device, vec![M, K], 0.0..1.0);
    let b = Tensor::random(&device, vec![K, N], 0.0..1.0);
    let mut c = Tensor::new(&device, vec![M, N]);
    let d = None; //Tensor::new(&device, vec![2, 3]);

    let cases = [
        (false, false, 1.0, 0.0),
        (true, false, 1.0, 0.0),
        (false, true, 1.0, 0.0),
    ];
    for (t_a, t_b, alpha, beta) in cases {
        println!("Running with transpose left: {t_a}, transpose right: {t_b}, alpha: {alpha}, beta: {beta}");

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            encode_gemm(
                &device, encoder, &a, &b, &mut c, &d, t_a, t_b, false, 1.0, 0.0, false,
            )
            .expect("Encoding failed");
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let total_time = start.elapsed();

        // Calculate GFLOPS
        // C <- alpha * AB + beta * C
        // Operations = 2(M * N * K)
        let avg_gflops =
            (ITERATIONS * (M * N * (2 * K - 1))) as f64 / (total_time.as_secs_f64() * 1e+9f64);

        println!("Avg GFLOPS: {}", avg_gflops);
        println!("Total time: {:#?}", total_time);
        println!()
    }
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
