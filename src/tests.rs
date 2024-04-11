use crate::MorseSmaleSolver;

#[pollster::test]
async fn construction_is_ok() {
    let try_construct = MorseSmaleSolver::new().await;
    assert!(try_construct.is_ok());
}

#[pollster::test]
async fn running_is_ok() {
    let input = vec![1, 2, 3, 4];
    let solver = MorseSmaleSolver::new().await.unwrap();
    let try_run = solver.run(input).await;
    assert!(try_run.is_ok());
}

#[pollster::test]
async fn simple_correctness() {
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

    assert_eq!(&formatted_steps[..4], &["0", "1", "7", "2"]);
}