use rgou_ai_core::{GameState, HeuristicAI, Player, AI, ml_ai::MLAI};
use std::time::Instant;

const QUICK_GAMES: usize = 5;
const PERFORMANCE_LIMIT_MS: u128 = 10;

#[test]
fn test_quick_ai_functionality() {
    println!("⚡ QUICK AI FUNCTIONALITY CHECK");
    println!("{}", "=".repeat(50));
    println!("Running essential tests with {} games each", QUICK_GAMES);
    println!();

    test_basic_ai_functionality();
    test_enhanced_evaluation_works();
    test_ai_can_make_moves();
    test_game_completion();
    test_ml_ai_weight_loading();
    test_ai_performance_limits();
    test_wasm_integration_safety();

    println!("✅ Quick check complete!");
}

fn test_basic_ai_functionality() {
    println!("🔍 Test 1: Basic AI Functionality");

    let mut ai = AI::new();
    let mut state = GameState::new();
    state.dice_roll = 1;

    let (best_move, evaluations) = ai.get_best_move(&state, 1);

    assert!(best_move.is_some(), "AI should find a valid move");
    assert!(
        !evaluations.is_empty(),
        "AI should provide move evaluations"
    );

    println!("  ✅ AI can find moves and provide evaluations");
}

fn test_enhanced_evaluation_works() {
    println!("🔍 Test 2: Enhanced Evaluation Function");

    let state = GameState::new();
    let score = state.evaluate();

    assert!(
        score.abs() < 1000,
        "Initial state should have reasonable score"
    );

    let mut state_with_pieces = GameState::new();
    state_with_pieces.player1_pieces[0].square = 0;
    state_with_pieces.player2_pieces[0].square = 4;

    let score_with_pieces = state_with_pieces.evaluate();
    assert!(
        score_with_pieces != score,
        "Different positions should have different scores"
    );

    println!("  ✅ Enhanced evaluation function works correctly");
}

fn test_ai_can_make_moves() {
    println!("🔍 Test 3: AI Can Make Valid Moves");

    let mut ai = AI::new();
    let mut state = GameState::new();
    state.dice_roll = 1;

    let (best_move, _) = ai.get_best_move(&state, 1);

    if let Some(move_index) = best_move {
        let result = state.make_move(move_index);
        assert!(result.is_ok(), "AI should suggest valid moves");
        println!("  ✅ AI suggested valid move: {}", move_index);
    } else {
        panic!("AI should find a valid move");
    }
}

fn test_game_completion() {
    println!("🔍 Test 4: Game Completion Test");

    let mut ai1 = AI::new();
    let mut ai2 = AI::new();
    let mut state = GameState::new();
    let mut moves = 0;
    let max_moves = 50;

    while !state.is_game_over() && moves < max_moves {
        state.dice_roll = 1;
        let current_ai = if state.current_player == Player::Player1 {
            &mut ai1
        } else {
            &mut ai2
        };

        let (best_move, _) = current_ai.get_best_move(&state, 1);

        if let Some(move_index) = best_move {
            state.make_move(move_index).unwrap();
            moves += 1;
        } else {
            state.current_player = state.current_player.opponent();
        }
    }

    assert!(moves > 0, "Game should progress");
    println!("  ✅ Game progressed {} moves", moves);
}

fn test_ml_ai_weight_loading() {
    println!("🔍 Test 5: ML AI Weight Loading & Persistence");

    let mut ml_ai = MLAI::new();

    let test_value_weights = vec![0.1; 1000];
    let test_policy_weights = vec![0.2; 1000];
    ml_ai.load_pretrained(&test_value_weights, &test_policy_weights);

    let mut state = GameState::new();
    state.dice_roll = 1;

    let start = Instant::now();
    let response = ml_ai.get_best_move(&state);
    let duration = start.elapsed();

    assert!(
        response.r#move.is_some(),
        "ML AI should find moves after weight loading"
    );
    assert!(
        duration.as_millis() < PERFORMANCE_LIMIT_MS,
        "ML AI should be fast: {}ms",
        duration.as_millis()
    );

    println!(
        "  ✅ ML AI loads weights and makes moves in {}ms",
        duration.as_millis()
    );
}

fn test_ai_performance_limits() {
    println!("🔍 Test 6: AI Performance Benchmarks");

    let mut ai = AI::new();
    let mut state = GameState::new();
    state.dice_roll = 1;

    let mut total_time = 0u128;
    let test_moves = 10;

    for _ in 0..test_moves {
        let start = Instant::now();
        let (best_move, _) = ai.get_best_move(&state, 1);
        let duration = start.elapsed();
        total_time += duration.as_millis();

        assert!(best_move.is_some(), "AI should find moves consistently");
        assert!(
            duration.as_millis() < PERFORMANCE_LIMIT_MS,
            "AI move too slow: {}ms",
            duration.as_millis()
        );

        if let Some(move_index) = best_move {
            state.make_move(move_index).unwrap();
        }
    }

    let avg_time = total_time / test_moves;
    assert!(
        avg_time < PERFORMANCE_LIMIT_MS,
        "Average AI time too slow: {}ms",
        avg_time
    );

    println!(
        "  ✅ AI performance: {}ms average (limit: {}ms)",
        avg_time, PERFORMANCE_LIMIT_MS
    );
}

fn test_wasm_integration_safety() {
    println!("🔍 Test 7: WASM Integration Safety");

    let mut ai = AI::new();
    let mut state = GameState::new();
    state.dice_roll = 1;

    let mut consecutive_moves = 0;
    let max_consecutive = 20;

    while consecutive_moves < max_consecutive && !state.is_game_over() {
        let (best_move, _) = ai.get_best_move(&state, 1);

        if let Some(move_index) = best_move {
            let result = state.make_move(move_index);
            assert!(result.is_ok(), "WASM AI should maintain state consistency");
            consecutive_moves += 1;
        } else {
            state.current_player = state.current_player.opponent();
        }
    }

    assert!(
        consecutive_moves > 0,
        "WASM AI should handle consecutive operations"
    );
    println!(
        "  ✅ WASM AI handled {} consecutive moves safely",
        consecutive_moves
    );
}

#[test]
fn test_quick_heuristic_vs_expectiminimax() {
    println!("🔍 Test 8: Quick Heuristic vs Expectiminimax");

    let mut heuristic_ai = HeuristicAI::new();
    let mut expectiminimax_ai = AI::new();

    let mut heuristic_wins = 0;
    let mut expectiminimax_wins = 0;

    for game in 0..QUICK_GAMES {
        let (winner, _) = play_quick_game(&mut heuristic_ai, &mut expectiminimax_ai, game % 2 == 0);

        if winner == Player::Player1 {
            heuristic_wins += 1;
        } else {
            expectiminimax_wins += 1;
        }
    }

    println!(
        "  Heuristic wins: {} ({}%)",
        heuristic_wins,
        (heuristic_wins as f64 / QUICK_GAMES as f64) * 100.0
    );
    println!(
        "  Expectiminimax wins: {} ({}%)",
        expectiminimax_wins,
        (expectiminimax_wins as f64 / QUICK_GAMES as f64) * 100.0
    );

    assert!(
        heuristic_wins + expectiminimax_wins == QUICK_GAMES,
        "All games should complete"
    );
    println!("  ✅ Quick comparison completed");
}

fn play_quick_game(
    heuristic_ai: &mut HeuristicAI,
    expectiminimax_ai: &mut AI,
    heuristic_plays_first: bool,
) -> (Player, usize) {
    let mut game_state = GameState::new();
    let mut moves_played = 0;
    let max_moves = 100;

    if !heuristic_plays_first {
        game_state.current_player = Player::Player2;
    }

    loop {
        let current_player = game_state.current_player;
        game_state.dice_roll = 1;

        if game_state.dice_roll == 0 {
            game_state.current_player = game_state.current_player.opponent();
            continue;
        }

        let best_move = if current_player == Player::Player1 {
            let (move_option, _) = heuristic_ai.get_best_move(&game_state);
            move_option
        } else {
            let (move_option, _) = expectiminimax_ai.get_best_move(&game_state, 1);
            move_option
        };

        if let Some(piece_index) = best_move {
            game_state.make_move(piece_index).unwrap();
            moves_played += 1;

            if game_state.is_game_over() {
                let p1_finished = game_state
                    .player1_pieces
                    .iter()
                    .filter(|p| p.square == 20)
                    .count();
                if p1_finished == 7 {
                    return (Player::Player1, moves_played);
                } else {
                    return (Player::Player2, moves_played);
                }
            }
        } else {
            game_state.current_player = game_state.current_player.opponent();
        }

        if moves_played >= max_moves {
            let p1_finished = game_state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let p2_finished = game_state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();

            if p1_finished > p2_finished {
                return (Player::Player1, moves_played);
            } else if p2_finished > p1_finished {
                return (Player::Player2, moves_played);
            } else {
                return (game_state.current_player, moves_played);
            }
        }
    }
}
