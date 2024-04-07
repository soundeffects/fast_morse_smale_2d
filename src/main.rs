use fast_morse_smale::MorseSmaleSolver;

async fn run() {
    let input = vec![1, 2, 3, 4];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let steps = solver.run(input.clone()).await.unwrap();

    let formatted_steps: Vec<String> = steps
        .iter()
        .map(|&number| match number {
            0xffffffff => "OVERFLOW".to_string(),
            _ => number.to_string(),
        })
        .collect();
    println!("Steps: [{}]", formatted_steps.join(", "));
    #[cfg(target_arch = "wasm32")]
    log::info!("Steps: [{}]", formatted_steps.join(", "));
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
