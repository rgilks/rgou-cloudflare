use rand::Rng;
use rgou_ai_core::genetic_ai::{GeneticAI, GeneticAlgorithm, HeuristicParams};
use rgou_ai_core::{GameState, Player};
use std::time::Instant;

fn main() {
    println!("🔬 GENETIC ALGORITHM EVOLUTION COMPARISON");
    println!("=========================================");
    println!();

    // Test different configurations
    let configurations = vec![
        ("Basic (10 games)", 20, 0.1, 3, 10),
        ("Enhanced (50 games)", 50, 0.05, 5, 50),
        ("High Precision (100 games)", 100, 0.02, 7, 100),
    ];

    let mut results = Vec::new();

    for (name, pop_size, mutation_rate, tournament_size, games_per_individual) in configurations {
        println!("🧬 Testing {} configuration:", name);
        println!(
            "  Population: {}, Mutation: {:.1}%, Tournament: {}, Games: {}",
            pop_size,
            mutation_rate * 100.0,
            tournament_size,
            games_per_individual
        );

        let ga = GeneticAlgorithm::new(
            pop_size,
            mutation_rate,
            tournament_size,
            games_per_individual,
        );

        let start_time = Instant::now();
        let best_params = ga.evolve(20); // Reduced generations for comparison
        let evolution_time = start_time.elapsed();

        // Test the evolved parameters
        let test_score = test_parameters(&best_params);

        results.push((name.to_string(), best_params, test_score, evolution_time));

        println!("  Evolution time: {:.2}s", evolution_time.as_secs_f64());
        println!("  Test score: {:.1}%", test_score * 100.0);
        println!();
    }

    // Compare results
    println!("📊 COMPARISON RESULTS:");
    println!(
        "{:<20} {:<15} {:<15} {:<15}",
        "Configuration", "Win Score", "Position Weight", "Test Score"
    );
    println!("{}", "-".repeat(70));

    for (name, params, test_score, _) in &results {
        println!(
            "{:<20} {:<15} {:<15} {:<15.1}%",
            name,
            params.win_score,
            params.position_weight,
            test_score * 100.0
        );
    }

    // Find best configuration
    let best_result = results
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    println!(
        "\n🏆 BEST CONFIGURATION: {} (Score: {:.1}%)",
        best_result.0,
        best_result.2 * 100.0
    );

    // Show parameter differences
    println!("\n📈 PARAMETER ANALYSIS:");
    let default_params = HeuristicParams::new();
    for (name, params, _, _) in &results {
        println!("\n{}:", name);
        println!(
            "  Position weight: {} → {} ({:+.1}%)",
            default_params.position_weight,
            params.position_weight,
            ((params.position_weight as f64 / default_params.position_weight as f64) - 1.0) * 100.0
        );
        println!(
            "  Rosette control: {} → {} ({:+.1}%)",
            default_params.rosette_chain_bonus,
            params.rosette_chain_bonus,
            ((params.rosette_chain_bonus as f64 / default_params.rosette_chain_bonus as f64)
                - 1.0)
                * 100.0
        );
        println!(
            "  Advancement bonus: {} → {} ({:+.1}%)",
            default_params.advancement_bonus,
            params.advancement_bonus,
            ((params.advancement_bonus as f64 / default_params.advancement_bonus as f64) - 1.0)
                * 100.0
        );
    }
}

fn test_parameters(params: &HeuristicParams) -> f64 {
    let mut genetic_ai = GeneticAI::new(params.clone());
    let mut heuristic_ai = rgou_ai_core::HeuristicAI::new();

    let mut wins = 0;
    let total_games = 20;

    for _ in 0..total_games {
        let mut game_state = GameState::new();
        let mut moves = 0;
        let max_moves = 200;

        while !game_state.is_game_over() && moves < max_moves {
            game_state.dice_roll = rand::thread_rng().gen_range(1..5);
            let valid_moves = game_state.get_valid_moves();

            if valid_moves.is_empty() {
                game_state.current_player = game_state.current_player.opponent();
                continue;
            }

            let best_move = if game_state.current_player == Player::Player1 {
                let (move_option, _) = genetic_ai.get_best_move(&game_state);
                move_option
            } else {
                let (move_option, _) = heuristic_ai.get_best_move(&game_state);
                move_option
            };

            if let Some(piece_index) = best_move {
                if game_state.make_move(piece_index).is_ok() {
                    moves += 1;
                }
            } else {
                game_state.current_player = game_state.current_player.opponent();
            }
        }

        // Determine winner
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
            wins += 1;
        }
    }

    wins as f64 / total_games as f64
}
