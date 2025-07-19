use rgou_ai_core::genetic_ai::{GeneticAI, HeuristicParams};
use rgou_ai_core::{roll_tetrahedral_dice, GameState, HeuristicAI, Player};

fn main() {
    println!("🐛 DEBUG GENETIC AI");
    println!("==================");

    // Test 1: Compare move selection
    println!("📊 TEST 1: Move Selection Comparison");
    println!("===================================");

    let mut state = GameState::new();
    state.dice_roll = 2; // Give a reasonable dice roll

    let default_params = HeuristicParams::new();
    let mut genetic_ai = GeneticAI::new(default_params);
    let mut heuristic_ai = HeuristicAI::new();

    println!("Initial state:");
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
    println!("  Dice roll: {}", state.dice_roll);

    let valid_moves = state.get_valid_moves();
    println!("  Valid moves: {:?}", valid_moves);

    if !valid_moves.is_empty() {
        let (genetic_move, genetic_evaluations) = genetic_ai.get_best_move(&state);
        let (heuristic_move, heuristic_evaluations) = heuristic_ai.get_best_move(&state);

        println!("\nGenetic AI move: {:?}", genetic_move);
        println!("Genetic AI evaluations:");
        for eval in &genetic_evaluations[..std::cmp::min(3, genetic_evaluations.len())] {
            println!(
                "  Piece {}: score {:.2} ({})",
                eval.piece_index, eval.score, eval.move_type
            );
        }

        println!("\nHeuristic AI move: {:?}", heuristic_move);
        println!("Heuristic AI evaluations:");
        for eval in &heuristic_evaluations[..std::cmp::min(3, heuristic_evaluations.len())] {
            println!(
                "  Piece {}: score {:.2} ({})",
                eval.piece_index, eval.score, eval.move_type
            );
        }

        // Test 2: Play a few moves and see what happens
        println!("\n📊 TEST 2: Play a Few Moves");
        println!("============================");

        let mut test_state = state.clone();
        let mut moves_played = 0;

        while !test_state.is_game_over() && moves_played < 10 {
            test_state.dice_roll = roll_tetrahedral_dice();
            let valid_moves = test_state.get_valid_moves();

            if valid_moves.is_empty() {
                test_state.current_player = test_state.current_player.opponent();
                continue;
            }

            let (best_move, _) = if test_state.current_player == Player::Player1 {
                genetic_ai.get_best_move(&test_state)
            } else {
                heuristic_ai.get_best_move(&test_state)
            };

            if let Some(move_idx) = best_move {
                println!(
                    "Move {}: Player{:?} moves piece {} (dice: {})",
                    moves_played + 1,
                    test_state.current_player,
                    move_idx,
                    test_state.dice_roll
                );

                if test_state.make_move(move_idx).is_ok() {
                    moves_played += 1;

                    let p1_finished = test_state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    let p2_finished = test_state
                        .player2_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    println!(
                        "  P1 finished: {}, P2 finished: {}",
                        p1_finished, p2_finished
                    );
                }
            } else {
                test_state.current_player = test_state.current_player.opponent();
            }
        }

        println!("\nFinal state after {} moves:", moves_played);
        println!(
            "  P1 finished: {}",
            test_state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count()
        );
        println!(
            "  P2 finished: {}",
            test_state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count()
        );
        println!("  Game over: {}", test_state.is_game_over());
    }

    // Test 3: Simple win rate test
    println!("\n📊 TEST 3: Simple Win Rate Test");
    println!("===============================");

    let mut genetic_wins = 0;
    let mut total_games = 0;

    for _ in 0..10 {
        let (winner, _) = play_single_game(&mut genetic_ai, &mut heuristic_ai, true);
        if winner == Player::Player1 {
            genetic_wins += 1;
        }
        total_games += 1;

        let (winner, _) = play_single_game(&mut genetic_ai, &mut heuristic_ai, false);
        if winner == Player::Player1 {
            genetic_wins += 1;
        }
        total_games += 1;
    }

    println!(
        "Genetic AI wins: {}/{} = {:.1}%",
        genetic_wins,
        total_games,
        (genetic_wins as f64 / total_games as f64) * 100.0
    );
}

fn play_single_game(
    ai1: &mut GeneticAI,
    ai2: &mut HeuristicAI,
    ai1_plays_first: bool,
) -> (Player, usize) {
    let mut state = GameState::new();
    let mut moves_played = 0;

    if !ai1_plays_first {
        state.current_player = Player::Player2;
    }

    while !state.is_game_over() && moves_played < 200 {
        state.dice_roll = roll_tetrahedral_dice();
        let valid_moves = state.get_valid_moves();

        if valid_moves.is_empty() {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let (best_move, _) = if state.current_player == Player::Player1 {
            ai1.get_best_move(&state)
        } else {
            ai2.get_best_move(&state)
        };

        if let Some(move_idx) = best_move {
            if state.make_move(move_idx).is_ok() {
                moves_played += 1;
            }
        } else {
            state.current_player = state.current_player.opponent();
        }
    }

    let winner = if state
        .player1_pieces
        .iter()
        .filter(|p| p.square == 20)
        .count()
        == rgou_ai_core::PIECES_PER_PLAYER
    {
        Player::Player1
    } else {
        Player::Player2
    };

    (winner, moves_played)
}
