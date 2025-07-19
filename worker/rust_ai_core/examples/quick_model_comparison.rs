use rgou_ai_core::{
    genetic_ai::{GeneticAI, HeuristicParams},
    HeuristicAI, AI, GameState, Player,
};
use std::collections::HashMap;
use rand::Rng;

fn main() {
    println!("⚔️  QUICK MODEL COMPARISON");
    println!("=========================");

    // Create evolved parameters from the quick evolution run
    let evolved_params = HeuristicParams {
        win_score: 11610,
        finished_piece_value: 1228,
        position_weight: 27,
        advancement_bonus: 11,
        rosette_safety_bonus: 18,
        rosette_chain_bonus: 7,
        capture_bonus: 37,
        vulnerability_penalty: 8,
        center_control_bonus: 5,
        piece_coordination_bonus: 3,
        blocking_bonus: 18,
        early_game_bonus: 5,
        late_game_urgency: 12,
        turn_order_bonus: 10,
        mobility_bonus: 5,
        attack_pressure_bonus: 9,
        defensive_structure_bonus: 3,
    };

    let mut genetic_ai = GeneticAI::new(evolved_params);
    let mut heuristic_ai = HeuristicAI::new();
    let mut expectiminimax_ai = AI::new();

    let games_per_matchup = 20; // Quick test
    let mut results = HashMap::new();

    println!("Playing {} games per matchup...", games_per_matchup);

    // Test Genetic AI vs Heuristic AI
    let (genetic_wins, heuristic_wins) = play_matchup(
        &mut genetic_ai,
        &mut heuristic_ai,
        games_per_matchup,
        "Genetic AI",
        "Heuristic AI",
    );
    results.insert("Genetic vs Heuristic", (genetic_wins, heuristic_wins));

    // Test Genetic AI vs Expectiminimax Depth 2
    let (genetic_wins, expectiminimax_wins) = play_matchup(
        &mut genetic_ai,
        &mut expectiminimax_ai,
        games_per_matchup,
        "Genetic AI",
        "Expectiminimax Depth 2",
    );
    results.insert("Genetic vs Expectiminimax", (genetic_wins, expectiminimax_wins));

    println!("\n📊 RESULTS:");
    println!("============");

    for (matchup, (ai1_wins, ai2_wins)) in &results {
        let total = ai1_wins + ai2_wins;
        let ai1_rate = (*ai1_wins as f64 / total as f64) * 100.0;
        let ai2_rate = (*ai2_wins as f64 / total as f64) * 100.0;

        println!(
            "{}: {} wins ({}%) vs {} wins ({}%)",
            matchup, ai1_wins, ai1_rate, ai2_wins, ai2_rate
        );
    }

    println!("\n🎯 Genetic AI Performance Summary:");
    let total_genetic_wins: usize = results.values().map(|(wins, _)| wins).sum();
    let total_games: usize = results.values().map(|(wins, losses)| wins + losses).sum();
    let overall_rate = (total_genetic_wins as f64 / total_games as f64) * 100.0;

    println!(
        "Total wins: {}/{} ({:.1}%)",
        total_genetic_wins, total_games, overall_rate
    );

    if overall_rate > 60.0 {
        println!("✅ Strong performance! Ready for longer evolution.");
    } else if overall_rate > 45.0 {
        println!("⚠️  Moderate performance. May need parameter tuning.");
    } else {
        println!("❌ Weak performance. Consider adjusting evolution parameters.");
    }
}

fn play_matchup(
    ai1: &mut GeneticAI,
    ai2: &mut dyn std::any::Any,
    games: usize,
    ai1_name: &str,
    ai2_name: &str,
) -> (usize, usize) {
    let mut ai1_wins = 0;
    let mut ai2_wins = 0;

    for game in 0..games {
        let mut state = GameState::new();

        while !state.is_game_over() {
            state.dice_roll = rand::thread_rng().gen_range(1..5);
            let valid_moves = state.get_valid_moves();

            if valid_moves.is_empty() {
                state.current_player = state.current_player.opponent();
                continue;
            }

            let best_move = if state.current_player == Player::Player1 {
                let (move_option, _) = ai1.get_best_move(&state);
                move_option
            } else {
                // Handle different AI types
                if let Some(heuristic_ai) = ai2.downcast_mut::<HeuristicAI>() {
                    let (move_option, _) = heuristic_ai.get_best_move(&state);
                    move_option
                } else if let Some(expectiminimax_ai) = ai2.downcast_mut::<AI>() {
                    let (move_option, _) = expectiminimax_ai.get_best_move(&state, 2);
                    move_option
                } else {
                    None
                }
            };

            if let Some(piece_index) = best_move {
                if valid_moves.contains(&piece_index) {
                    state.make_move(piece_index).unwrap();
                }
            } else {
                state.current_player = state.current_player.opponent();
            }
        }

        // Determine winner
        let p1_finished = state
            .player1_pieces
            .iter()
            .filter(|p| p.square == 20)
            .count();
        if p1_finished == 7 {
            ai1_wins += 1;
        } else {
            ai2_wins += 1;
        }

        if (game + 1) % 10 == 0 {
            println!("  Game {}: {} wins, {} wins", game + 1, ai1_wins, ai2_wins);
        }
    }

    (ai1_wins, ai2_wins)
}
