use rgou_ai_core::genetic_ai::{GeneticAI, HeuristicParams};
use rgou_ai_core::{GameState, HeuristicAI, Player};

fn main() {
    println!("🔍 EVALUATION FUNCTION COMPARISON");
    println!("=================================");

    // Create a test state
    let mut state = GameState::new();
    state.dice_roll = 2;

    // Test with default parameters
    let default_params = HeuristicParams::new();
    let genetic_ai = GeneticAI::new(default_params.clone());
    let heuristic_ai = HeuristicAI::new();

    println!("Test state:");
    println!(
        "  Player1 pieces: {:?}",
        state
            .player1_pieces
            .iter()
            .map(|p| p.square)
            .collect::<Vec<_>>()
    );
    println!(
        "  Player2 pieces: {:?}",
        state
            .player2_pieces
            .iter()
            .map(|p| p.square)
            .collect::<Vec<_>>()
    );
    println!("  Current player: {:?}", state.current_player);
    println!();

    // Compare evaluations for each possible move
    let valid_moves = state.get_valid_moves();
    println!("Move evaluations:");
    println!("Piece | Genetic Score | Heuristic Score | Difference");
    println!("------|---------------|-----------------|------------");

    for &piece_index in &valid_moves[..std::cmp::min(5, valid_moves.len())] {
        let mut test_state = state.clone();
        if test_state.make_move(piece_index).is_ok() {
            let genetic_score = genetic_ai.evaluate_with_params(&test_state, &default_params);
            let heuristic_score = test_state.evaluate();

            println!(
                "{:5} | {:13} | {:15} | {:10}",
                piece_index,
                genetic_score,
                heuristic_score,
                genetic_score - heuristic_score
            );
        }
    }

    // Test with a more advanced state
    println!("\n📊 ADVANCED STATE TEST");
    println!("======================");

    let mut advanced_state = GameState::new();
    // Move some pieces to create a more interesting position
    advanced_state.player1_pieces[0].square = 3; // On the board
    advanced_state.player1_pieces[1].square = 7; // On a rosette
    advanced_state.player2_pieces[0].square = 4; // On the board
    advanced_state.player2_pieces[1].square = 13; // On a rosette
    advanced_state.dice_roll = 2;

    // Update the board to reflect piece positions
    advanced_state.board[3] = Some(rgou_ai_core::PiecePosition {
        square: 3,
        player: Player::Player1,
    });
    advanced_state.board[7] = Some(rgou_ai_core::PiecePosition {
        square: 7,
        player: Player::Player1,
    });
    advanced_state.board[4] = Some(rgou_ai_core::PiecePosition {
        square: 4,
        player: Player::Player2,
    });
    advanced_state.board[13] = Some(rgou_ai_core::PiecePosition {
        square: 13,
        player: Player::Player2,
    });

    println!("Advanced state:");
    println!(
        "  Player1 pieces: {:?}",
        advanced_state
            .player1_pieces
            .iter()
            .map(|p| p.square)
            .collect::<Vec<_>>()
    );
    println!(
        "  Player2 pieces: {:?}",
        advanced_state
            .player2_pieces
            .iter()
            .map(|p| p.square)
            .collect::<Vec<_>>()
    );
    println!("  Current player: {:?}", advanced_state.current_player);
    println!();

    let valid_moves = advanced_state.get_valid_moves();
    println!("Advanced state move evaluations:");
    println!("Piece | Genetic Score | Heuristic Score | Difference");
    println!("------|---------------|-----------------|------------");

    for &piece_index in &valid_moves[..std::cmp::min(5, valid_moves.len())] {
        let mut test_state = advanced_state.clone();
        if test_state.make_move(piece_index).is_ok() {
            let genetic_score = genetic_ai.evaluate_with_params(&test_state, &default_params);
            let heuristic_score = test_state.evaluate();

            println!(
                "{:5} | {:13} | {:15} | {:10}",
                piece_index,
                genetic_score,
                heuristic_score,
                genetic_score - heuristic_score
            );
        }
    }

    // Test parameter differences
    println!("\n📊 PARAMETER COMPARISON");
    println!("======================");

    println!("Default Genetic Parameters:");
    println!("  win_score: {}", default_params.win_score);
    println!(
        "  finished_piece_value: {}",
        default_params.finished_piece_value
    );
    println!("  position_weight: {}", default_params.position_weight);
    println!("  advancement_bonus: {}", default_params.advancement_bonus);
    println!(
        "  rosette_safety_bonus: {}",
        default_params.rosette_safety_bonus
    );
    println!("  capture_bonus: {}", default_params.capture_bonus);
    println!(
        "  vulnerability_penalty: {}",
        default_params.vulnerability_penalty
    );

    // Check if there's a fundamental issue with the evaluation
    println!("\n📊 FUNDAMENTAL TEST");
    println!("==================");

    let mut test_state = GameState::new();
    test_state.player1_pieces[0].square = 20; // Player1 has one finished piece
    test_state.player2_pieces[0].square = 20; // Player2 has one finished piece

    let genetic_score = genetic_ai.evaluate_with_params(&test_state, &default_params);
    let heuristic_score = test_state.evaluate();

    println!("Equal position (both have 1 finished piece):");
    println!("  Genetic score: {}", genetic_score);
    println!("  Heuristic score: {}", heuristic_score);
    println!("  Difference: {}", genetic_score - heuristic_score);

    // Test winning position
    let mut winning_state = GameState::new();
    winning_state.player1_pieces[0].square = 20;
    winning_state.player1_pieces[1].square = 20;
    winning_state.player1_pieces[2].square = 20;
    winning_state.player1_pieces[3].square = 20;
    winning_state.player1_pieces[4].square = 20;
    winning_state.player1_pieces[5].square = 20;
    winning_state.player1_pieces[6].square = 20;

    let genetic_score = genetic_ai.evaluate_with_params(&winning_state, &default_params);
    let heuristic_score = winning_state.evaluate();

    println!("\nPlayer1 winning position:");
    println!("  Genetic score: {}", genetic_score);
    println!("  Heuristic score: {}", heuristic_score);
    println!("  Difference: {}", genetic_score - heuristic_score);
}
