use fast_morse_smale_2d::MorseSmaleSolver;
use pollster::test as async_test;

#[async_test]
async fn million_stress() {
    let samples = vec![1.0; 1_000_000];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let result = solver.with_samples(samples, 1000).await;
    assert!(result.is_ok());
}
