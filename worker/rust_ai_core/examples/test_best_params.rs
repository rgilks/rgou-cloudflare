use rgou_ai_core::genetic_ai::{GeneticAI, HeuristicParams};
use rgou_ai_core::{roll_tetrahedral_dice, GameState, HeuristicAI, Player};

fn main() {
    println!("🧪 TEST BEST GENETIC PARAMETERS");
    println!("===============================");

    // Load the best parameters
    let best_params = HeuristicParams::from_file("best_genetic_params.json").unwrap();
    let default_params = HeuristicParams::new();

    println!("Testing parameters:");
    println!(
        "  Best params: win_score={}, finished_piece_value={}, position_weight={}",
        best_params.win_score, best_params.finished_piece_value, best_params.position_weight
    );
    println!(
        "  Default params: win_score={}, finished_piece_value={}, position_weight={}",
        default_params.win_score,
        default_params.finished_piece_value,
        default_params.position_weight
    );
    println!();

    // Test best params vs heuristic
    let mut best_ai = GeneticAI::new(best_params.clone());
    let mut default_ai = GeneticAI::new(default_params.clone());
    let mut heuristic_ai = HeuristicAI::new();

    println!("📊 TEST 1: Best vs Heuristic");
    println!("============================");
    let (best_wins, best_total) = play_multiple_games(&mut best_ai, &mut heuristic_ai, 50);
    let best_win_rate = best_wins as f64 / best_total as f64;
    println!(
        "Best params vs Heuristic: {}/{} = {:.1}%",
        best_wins,
        best_total,
        best_win_rate * 100.0
    );

    println!("\n📊 TEST 2: Default vs Heuristic");
    println!("===============================");
    let (default_wins, default_total) = play_multiple_games(&mut default_ai, &mut heuristic_ai, 50);
    let default_win_rate = default_wins as f64 / default_total as f64;
    println!(
        "Default params vs Heuristic: {}/{} = {:.1}%",
        default_wins,
        default_total,
        default_win_rate * 100.0
    );

    println!("\n📊 TEST 3: Best vs Default");
    println!("==========================");
    let (best_vs_default_wins, best_vs_default_total) =
        play_multiple_games_genetic(&mut best_ai, &mut default_ai, 50);
    let best_vs_default_rate = best_vs_default_wins as f64 / best_vs_default_total as f64;
    println!(
        "Best vs Default: {}/{} = {:.1}%",
        best_vs_default_wins,
        best_vs_default_total,
        best_vs_default_rate * 100.0
    );

    println!("\n📊 SUMMARY");
    println!("===========");
    println!(
        "Best params performance: {:.1}% vs Heuristic",
        best_win_rate * 100.0
    );
    println!(
        "Default params performance: {:.1}% vs Heuristic",
        default_win_rate * 100.0
    );
    println!("Best vs Default: {:.1}%", best_vs_default_rate * 100.0);

    if best_win_rate > default_win_rate {
        println!("✅ Best parameters are better than default!");
    } else {
        println!("❌ Best parameters are not better than default");
    }

    if best_win_rate > 0.4 {
        println!("✅ Best parameters perform reasonably well!");
    } else {
        println!("❌ Best parameters perform poorly");
    }
}

fn play_multiple_games(ai1: &mut GeneticAI, ai2: &mut HeuristicAI, num_games: u32) -> (u32, u32) {
    let mut wins = 0;
    let mut total = 0;

    for _ in 0..num_games {
        let (winner, _) = play_single_game(ai1, ai2, true);
        if winner == Player::Player1 {
            wins += 1;
        }
        total += 1;

        let (winner, _) = play_single_game(ai1, ai2, false);
        if winner == Player::Player1 {
            wins += 1;
        }
        total += 1;
    }

    (wins, total)
}

fn play_multiple_games_genetic(
    ai1: &mut GeneticAI,
    ai2: &mut GeneticAI,
    num_games: u32,
) -> (u32, u32) {
    let mut wins = 0;
    let mut total = 0;

    for _ in 0..num_games {
        let (winner, _) = play_single_game_genetic(ai1, ai2, true);
        if winner == Player::Player1 {
            wins += 1;
        }
        total += 1;

        let (winner, _) = play_single_game_genetic(ai1, ai2, false);
        if winner == Player::Player1 {
            wins += 1;
        }
        total += 1;
    }

    (wins, total)
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

fn play_single_game_genetic(
    ai1: &mut GeneticAI,
    ai2: &mut GeneticAI,
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
