use fast_morse_smale_2d::MorseSmaleSolver;
use pollster::test as async_test;

#[async_test]
async fn simple_correctness() {
    let samples = vec![1., 2., 3., 4.];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let solution = solver.with_samples(samples, 2).await.unwrap();
    assert_eq!(
        solution,
        vec![1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}
