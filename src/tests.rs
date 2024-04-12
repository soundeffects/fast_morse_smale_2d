// ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀░░░░█▀▄░█▀▀
// ░░█░░█▀▀░▀▀█░░█░░▀▀█░░░░█▀▄░▀▀█
// ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░░▀░▀░▀▀▀
//
// This file defines tests which check for errors and ensure correctness for the library.

use crate::MorseSmaleSolver;
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

#[async_test]
async fn million_stress() {
    let samples = vec![1.0; 1_000_000];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let result = solver.with_samples(samples, 1000).await;
    assert!(result.is_ok());
}
