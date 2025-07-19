use rgou_ai_core::{GameState, GeneticAI, HeuristicParams};

#[test]
fn test_enhanced_evaluation_vs_genetic_ai() {
    // Load the super-evolved parameters
    let params = HeuristicParams::from_file("super_evolved_params.json").unwrap();
    let genetic_ai = GeneticAI::new(params);

    // Create test positions
    let test_positions = create_test_positions();

    for (i, position) in test_positions.iter().enumerate() {
        println!("Testing position {}", i + 1);

        // Get evaluation from enhanced EMM-3
        let enhanced_score = position.evaluate();

        // Get evaluation from genetic AI
        let genetic_score = genetic_ai.evaluate_with_params(position, genetic_ai.get_params());

        println!("  Enhanced EMM-3 score: {}", enhanced_score);
        println!("  Genetic AI score: {}", genetic_score);
        println!("  Difference: {}", (enhanced_score - genetic_score).abs());

        // The scores should be reasonably close (within 20% or 1000 points)
        let tolerance = (enhanced_score.abs() * 20 / 100).max(1000);
        assert!(
            (enhanced_score - genetic_score).abs() <= tolerance,
            "Position {}: Enhanced score {} vs Genetic score {} (tolerance: {})",
            i + 1,
            enhanced_score,
            genetic_score,
            tolerance
        );
    }
}

#[test]
fn test_enhanced_evaluation_consistency() {
    let test_positions = create_test_positions();

    for (i, position) in test_positions.iter().enumerate() {
        let score1 = position.evaluate();
        let score2 = position.evaluate();

        assert_eq!(
            score1,
            score2,
            "Position {}: Evaluation not consistent ({} vs {})",
            i + 1,
            score1,
            score2
        );
    }
}

#[test]
fn test_enhanced_evaluation_sensitivity() {
    let mut base_state = GameState::new();

    // Test that small changes produce different evaluations
    let base_score = base_state.evaluate();

    // Add a piece to a rosette
    base_state.player1_pieces[0].square = 0;
    let rosette_score = base_state.evaluate();
    assert_ne!(base_score, rosette_score);

    // Add a vulnerable piece
    base_state.player1_pieces[1].square = 5;
    let vulnerable_score = base_state.evaluate();
    assert_ne!(rosette_score, vulnerable_score);

    // Add coordinated pieces
    base_state.player1_pieces[2].square = 6;
    let coordinated_score = base_state.evaluate();
    assert_ne!(vulnerable_score, coordinated_score);

    println!("Base score: {}", base_score);
    println!("Rosette score: {}", rosette_score);
    println!("Vulnerable score: {}", vulnerable_score);
    println!("Coordinated score: {}", coordinated_score);
}

fn create_test_positions() -> Vec<GameState> {
    let mut positions = Vec::new();

    // Position 1: Initial state
    positions.push(GameState::new());

    // Position 2: Early game with pieces on board
    let mut early_game = GameState::new();
    early_game.player1_pieces[0].square = 0; // Rosette
    early_game.player2_pieces[0].square = 4; // Center
    positions.push(early_game);

    // Position 3: Mid game with tactical opportunities
    let mut mid_game = GameState::new();
    mid_game.player1_pieces[0].square = 0; // Rosette
    mid_game.player1_pieces[1].square = 5; // Near opponent
    mid_game.player2_pieces[0].square = 4; // Center
    mid_game.player2_pieces[1].square = 6; // Coordinated
    positions.push(mid_game);

    // Position 4: Late game with finished pieces
    let mut late_game = GameState::new();
    late_game.player1_pieces[0].square = 20; // Finished
    late_game.player1_pieces[1].square = 20; // Finished
    late_game.player1_pieces[2].square = 13; // Near finish
    late_game.player2_pieces[0].square = 20; // Finished
    late_game.player2_pieces[1].square = 15; // Rosette
    positions.push(late_game);

    // Position 5: Complex tactical position
    let mut complex = GameState::new();
    complex.player1_pieces[0].square = 0; // Rosette
    complex.player1_pieces[1].square = 7; // Rosette
    complex.player1_pieces[2].square = 5; // Vulnerable
    complex.player2_pieces[0].square = 4; // Center
    complex.player2_pieces[1].square = 6; // Coordinated
    complex.player2_pieces[2].square = 8; // Coordinated
    positions.push(complex);

    positions
}
