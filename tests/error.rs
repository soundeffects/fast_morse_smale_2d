use fast_morse_smale_2d::MorseSmaleSolver;
use pollster::test as async_test;

#[async_test]
async fn construction_is_ok() {
    let solver = MorseSmaleSolver::new().await;
    assert!(solver.is_ok());
}

#[async_test]
async fn running_is_ok() {
    let samples = vec![1., 2., 3., 4.];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let result = solver.with_samples(samples, 2).await;
    assert!(result.is_ok());
}
