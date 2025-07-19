use rgou_ai_core::genetic_ai::{GeneticAI, GeneticAlgorithm, HeuristicParams};
use std::time::{Duration, Instant};

fn main() {
    println!("🧬 GENETIC AI PARAMETER REOPTIMIZATION");
    println!("=====================================");
    println!("Reoptimizing parameters after dice probability fix");
    println!("Target duration: 15 minutes");
    println!();

    let start_time = Instant::now();
    let target_duration = Duration::from_secs(15 * 60); // 15 minutes

    // Configure genetic algorithm for 15-minute run
    let ga = GeneticAlgorithm::new(
        30,   // Population size (increased for better exploration)
        0.05, // Mutation rate (slightly higher for more exploration)
        5,    // Tournament size
        25,   // Games per individual (increased for more reliable evaluation)
    );

    println!("Configuration:");
    println!("  Population: 30 individuals");
    println!("  Mutation rate: 5%");
    println!("  Tournament size: 5");
    println!("  Games per individual: 25");
    println!("  Target duration: 15 minutes");
    println!();

    let mut generation = 0;
    let mut best_params = HeuristicParams::new();
    let mut best_fitness = 0.0;

    println!("Starting evolution...");
    println!("Generation | Best Fitness | Win Rate | Time Elapsed");
    println!("----------|--------------|----------|-------------");

    loop {
        let generation_start = Instant::now();
        generation += 1;

        // Evolve one generation
        let evolved_params = ga.evolve(1);

        // Evaluate the best individual from this generation
        let mut test_ai = GeneticAI::new(evolved_params.clone());
        let mut heuristic_ai = rgou_ai_core::HeuristicAI::new();

        let (wins, total) = play_multiple_games(&mut test_ai, &mut heuristic_ai, 50);
        let win_rate = wins as f64 / total as f64;
        let fitness = win_rate * 100.0;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_params = evolved_params.clone();
        }

        let elapsed = start_time.elapsed();
        let _generation_time = generation_start.elapsed();

        println!(
            "{:9} | {:12.2} | {:8.1}% | {:11.1}s",
            generation,
            fitness,
            win_rate * 100.0,
            elapsed.as_secs_f64()
        );

        // Check if we've reached the time limit
        if elapsed >= target_duration {
            println!("\n⏰ Time limit reached! Evolution complete.");
            break;
        }

        // Also check if we've achieved excellent performance
        if fitness > 75.0 {
            println!("\n🎯 Excellent performance achieved! Evolution complete.");
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "\n✅ Evolution completed in {:.2} minutes",
        total_time.as_secs_f64() / 60.0
    );
    println!("🏆 Best fitness achieved: {:.2}%", best_fitness);
    println!("📊 Total generations: {}", generation);

    // Save the best parameters
    save_best_parameters(&best_params, best_fitness, generation, total_time);

    // Update the main library constants
    update_library_constants(&best_params);

    println!("\n💾 Parameters saved and library updated!");
}

fn play_multiple_games(
    ai1: &mut GeneticAI,
    ai2: &mut rgou_ai_core::HeuristicAI,
    num_games: u32,
) -> (u32, u32) {
    let mut wins = 0;
    let mut total = 0;

    for _ in 0..num_games {
        let (winner, _) = play_single_game(ai1, ai2, true);
        if winner == rgou_ai_core::Player::Player1 {
            wins += 1;
        }
        total += 1;

        let (winner, _) = play_single_game(ai1, ai2, false);
        if winner == rgou_ai_core::Player::Player1 {
            wins += 1;
        }
        total += 1;
    }

    (wins, total)
}

fn play_single_game(
    ai1: &mut GeneticAI,
    ai2: &mut rgou_ai_core::HeuristicAI,
    ai1_plays_first: bool,
) -> (rgou_ai_core::Player, usize) {
    let mut state = rgou_ai_core::GameState::new();
    let mut moves_played = 0;

    if !ai1_plays_first {
        state.current_player = rgou_ai_core::Player::Player2;
    }

    while !state.is_game_over() && moves_played < 200 {
        state.dice_roll = rgou_ai_core::roll_tetrahedral_dice();
        let valid_moves = state.get_valid_moves();

        if valid_moves.is_empty() {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let (best_move, _) = if state.current_player == rgou_ai_core::Player::Player1 {
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
        rgou_ai_core::Player::Player1
    } else {
        rgou_ai_core::Player::Player2
    };

    (winner, moves_played)
}

fn save_best_parameters(
    params: &HeuristicParams,
    fitness: f64,
    generations: u32,
    duration: Duration,
) {
    // Save as JSON
    let json_params = serde_json::to_string_pretty(&params).unwrap();
    std::fs::write("best_genetic_params.json", json_params).unwrap();
    println!("💾 Best parameters saved to: best_genetic_params.json");

    // Save detailed results
    let mut summary = String::new();
    summary.push_str("# Genetic AI Reoptimization Results\n\n");
    summary.push_str(&format!(
        "Optimization completed: {}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    ));
    summary.push_str(&format!(
        "Duration: {:.2} minutes\n",
        duration.as_secs_f64() / 60.0
    ));
    summary.push_str(&format!("Generations: {}\n", generations));
    summary.push_str(&format!("Best fitness: {:.2}%\n\n", fitness));

    summary.push_str("## Optimized Parameters\n\n");
    summary.push_str("```json\n");
    summary.push_str(&serde_json::to_string_pretty(&params).unwrap());
    summary.push_str("\n```\n\n");

    summary.push_str("## Parameter Changes from Previous Best\n\n");
    summary.push_str("| Parameter | Previous | New | Change |\n");
    summary.push_str("|-----------|----------|-----|--------|\n");

    // Load previous best parameters for comparison
    if let Ok(prev_content) = std::fs::read_to_string("best_genetic_params.json") {
        if let Ok(prev_params) = serde_json::from_str::<HeuristicParams>(&prev_content) {
            summary.push_str(&format!(
                "| win_score | {} | {} | {:+.0} |\n",
                prev_params.win_score,
                params.win_score,
                params.win_score - prev_params.win_score
            ));
            summary.push_str(&format!(
                "| finished_piece_value | {} | {} | {:+.0} |\n",
                prev_params.finished_piece_value,
                params.finished_piece_value,
                params.finished_piece_value - prev_params.finished_piece_value
            ));
            summary.push_str(&format!(
                "| position_weight | {} | {} | {:+.0} |\n",
                prev_params.position_weight,
                params.position_weight,
                params.position_weight - prev_params.position_weight
            ));
            summary.push_str(&format!(
                "| advancement_bonus | {} | {} | {:+.0} |\n",
                prev_params.advancement_bonus,
                params.advancement_bonus,
                params.advancement_bonus - prev_params.advancement_bonus
            ));
            summary.push_str(&format!(
                "| rosette_safety_bonus | {} | {} | {:+.0} |\n",
                prev_params.rosette_safety_bonus,
                params.rosette_safety_bonus,
                params.rosette_safety_bonus - prev_params.rosette_safety_bonus
            ));
            summary.push_str(&format!(
                "| rosette_chain_bonus | {} | {} | {:+.0} |\n",
                prev_params.rosette_chain_bonus,
                params.rosette_chain_bonus,
                params.rosette_chain_bonus - prev_params.rosette_chain_bonus
            ));
            summary.push_str(&format!(
                "| capture_bonus | {} | {} | {:+.0} |\n",
                prev_params.capture_bonus,
                params.capture_bonus,
                params.capture_bonus - prev_params.capture_bonus
            ));
            summary.push_str(&format!(
                "| vulnerability_penalty | {} | {} | {:+.0} |\n",
                prev_params.vulnerability_penalty,
                params.vulnerability_penalty,
                params.vulnerability_penalty - prev_params.vulnerability_penalty
            ));
            summary.push_str(&format!(
                "| center_control_bonus | {} | {} | {:+.0} |\n",
                prev_params.center_control_bonus,
                params.center_control_bonus,
                params.center_control_bonus - prev_params.center_control_bonus
            ));
            summary.push_str(&format!(
                "| piece_coordination_bonus | {} | {} | {:+.0} |\n",
                prev_params.piece_coordination_bonus,
                params.piece_coordination_bonus,
                params.piece_coordination_bonus - prev_params.piece_coordination_bonus
            ));
            summary.push_str(&format!(
                "| blocking_bonus | {} | {} | {:+.0} |\n",
                prev_params.blocking_bonus,
                params.blocking_bonus,
                params.blocking_bonus - prev_params.blocking_bonus
            ));
            summary.push_str(&format!(
                "| early_game_bonus | {} | {} | {:+.0} |\n",
                prev_params.early_game_bonus,
                params.early_game_bonus,
                params.early_game_bonus - prev_params.early_game_bonus
            ));
            summary.push_str(&format!(
                "| late_game_urgency | {} | {} | {:+.0} |\n",
                prev_params.late_game_urgency,
                params.late_game_urgency,
                params.late_game_urgency - prev_params.late_game_urgency
            ));
            summary.push_str(&format!(
                "| turn_order_bonus | {} | {} | {:+.0} |\n",
                prev_params.turn_order_bonus,
                params.turn_order_bonus,
                params.turn_order_bonus - prev_params.turn_order_bonus
            ));
            summary.push_str(&format!(
                "| mobility_bonus | {} | {} | {:+.0} |\n",
                prev_params.mobility_bonus,
                params.mobility_bonus,
                params.mobility_bonus - prev_params.mobility_bonus
            ));
            summary.push_str(&format!(
                "| attack_pressure_bonus | {} | {} | {:+.0} |\n",
                prev_params.attack_pressure_bonus,
                params.attack_pressure_bonus,
                params.attack_pressure_bonus - prev_params.attack_pressure_bonus
            ));
            summary.push_str(&format!(
                "| defensive_structure_bonus | {} | {} | {:+.0} |\n",
                prev_params.defensive_structure_bonus,
                params.defensive_structure_bonus,
                params.defensive_structure_bonus - prev_params.defensive_structure_bonus
            ));
        }
    }

    std::fs::write("genetic_reoptimization_results.md", summary).unwrap();
    println!("📝 Detailed results saved to: genetic_reoptimization_results.md");
}

fn update_library_constants(params: &HeuristicParams) {
    // Read the current lib.rs file
    let lib_content = std::fs::read_to_string("../src/lib.rs").unwrap_or_default();

    // Create the new constants section
    let new_constants = format!(
        "const WIN_SCORE: i32 = {};\nconst FINISHED_PIECE_VALUE: i32 = {};\nconst POSITION_WEIGHT: i32 = {};\nconst SAFETY_BONUS: i32 = {};\nconst ROSETTE_CONTROL_BONUS: i32 = {};\nconst ADVANCEMENT_BONUS: i32 = {};\nconst CAPTURE_BONUS: i32 = {};\nconst CENTER_LANE_BONUS: i32 = {};\nconst VULNERABILITY_PENALTY: i32 = {};\nconst PIECE_COORDINATION_BONUS: i32 = {};\nconst BLOCKING_BONUS: i32 = {};\nconst EARLY_GAME_BONUS: i32 = {};\nconst LATE_GAME_URGENCY: i32 = {};\nconst TURN_ORDER_BONUS: i32 = {};\nconst MOBILITY_BONUS: i32 = {};\nconst ATTACK_PRESSURE_BONUS: i32 = {};\nconst DEFENSIVE_STRUCTURE_BONUS: i32 = {};",
        params.win_score,
        params.finished_piece_value,
        params.position_weight,
        params.rosette_safety_bonus,
        params.rosette_chain_bonus,
        params.advancement_bonus,
        params.capture_bonus,
        params.center_control_bonus,
        params.vulnerability_penalty,
        params.piece_coordination_bonus,
        params.blocking_bonus,
        params.early_game_bonus,
        params.late_game_urgency,
        params.turn_order_bonus,
        params.mobility_bonus,
        params.attack_pressure_bonus,
        params.defensive_structure_bonus
    );

    // Replace the constants section in lib.rs
    let updated_content = lib_content.replace(
        "const WIN_SCORE: i32 = 16149;\nconst FINISHED_PIECE_VALUE: i32 = 813;\nconst POSITION_WEIGHT: i32 = 20;\nconst SAFETY_BONUS: i32 = 28;\nconst ROSETTE_CONTROL_BONUS: i32 = 28;\nconst ADVANCEMENT_BONUS: i32 = 13;\nconst CAPTURE_BONUS: i32 = 43;\nconst CENTER_LANE_BONUS: i32 = 20;\nconst VULNERABILITY_PENALTY: i32 = 14;\nconst PIECE_COORDINATION_BONUS: i32 = 3;\nconst BLOCKING_BONUS: i32 = 18;\nconst EARLY_GAME_BONUS: i32 = 14;\nconst LATE_GAME_URGENCY: i32 = 30;\nconst TURN_ORDER_BONUS: i32 = 11;\nconst MOBILITY_BONUS: i32 = 6;\nconst ATTACK_PRESSURE_BONUS: i32 = 9;\nconst DEFENSIVE_STRUCTURE_BONUS: i32 = 7;",
        &new_constants
    );

    std::fs::write("../src/lib.rs", updated_content).unwrap();
    println!("🔧 Library constants updated in src/lib.rs");
}
