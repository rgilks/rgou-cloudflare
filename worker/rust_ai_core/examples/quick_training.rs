use rgou_ai_core::genetic_ai::{GeneticAI, GeneticAlgorithm, HeuristicParams};
use rgou_ai_core::{roll_tetrahedral_dice, GameState, HeuristicAI, Player};
use std::time::{Duration, Instant};

fn main() {
    println!("🚀 QUICK GENETIC TRAINING");
    println!("========================");
    println!("3-minute optimization run");
    println!();

    let start_time = Instant::now();
    let target_duration = Duration::from_secs(3 * 60); // 3 minutes

    // Start with current best parameters if available
    let mut best_params =
        HeuristicParams::from_file("best_genetic_params.json").unwrap_or(HeuristicParams::new());
    let mut best_fitness = 0.0;

    // Test current best against heuristic
    let mut test_ai = GeneticAI::new(best_params.clone());
    let mut heuristic_ai = HeuristicAI::new();
    let (wins, total) = play_multiple_games(&mut test_ai, &mut heuristic_ai, 20);
    best_fitness = wins as f64 / total as f64 * 100.0;

    println!("Current best parameters: {:.1}% win rate", best_fitness);

    // Configure genetic algorithm
    let ga = GeneticAlgorithm::new(
        20,   // Population size
        0.08, // Mutation rate
        4,    // Tournament size
        20,   // Games per individual
    );

    let mut generation = 0;
    println!("\nStarting evolution...");
    println!("Generation | Fitness | Win Rate | Time");
    println!("----------|---------|----------|------");

    loop {
        let generation_start = Instant::now();
        generation += 1;

        // Evolve one generation
        let evolved_params = ga.evolve(1);

        // Evaluate against heuristic AI
        let mut test_ai = GeneticAI::new(evolved_params.clone());
        let mut heuristic_ai = HeuristicAI::new();

        let (wins, total) = play_multiple_games(&mut test_ai, &mut heuristic_ai, 30);
        let win_rate = wins as f64 / total as f64;
        let fitness = win_rate * 100.0;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_params = evolved_params.clone();
            println!("🎯 NEW BEST: {:.1}% win rate!", fitness);
        }

        let elapsed = start_time.elapsed();

        println!(
            "{:9} | {:7.1} | {:8.1}% | {:4.1}s",
            generation,
            fitness,
            win_rate * 100.0,
            elapsed.as_secs_f64()
        );

        // Check time limit
        if elapsed >= target_duration {
            println!("\n⏰ Time limit reached!");
            break;
        }

        // Check if we've achieved good performance
        if fitness > 55.0 {
            println!("\n🎯 Good performance achieved!");
            break;
        }

        // Stop if we're not improving after many generations
        if generation > 15 && best_fitness < 45.0 {
            println!("\n⚠️  Poor performance - stopping early");
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "\n✅ Training completed in {:.2} minutes",
        total_time.as_secs_f64() / 60.0
    );
    println!("🏆 Best fitness achieved: {:.1}%", best_fitness);
    println!("📊 Total generations: {}", generation);

    // Save results
    let json_params = serde_json::to_string_pretty(&best_params).unwrap();
    std::fs::write("quick_training_best_params.json", json_params).unwrap();
    println!("💾 Best parameters saved to: quick_training_best_params.json");

    // Final verification
    println!("\n🔍 FINAL VERIFICATION");
    println!("====================");

    let mut final_ai = GeneticAI::new(best_params.clone());
    let mut heuristic_ai = HeuristicAI::new();

    let (final_wins, final_total) = play_multiple_games(&mut final_ai, &mut heuristic_ai, 50);
    let final_win_rate = final_wins as f64 / final_total as f64;

    println!(
        "Final verification: {}/{} = {:.1}%",
        final_wins,
        final_total,
        final_win_rate * 100.0
    );

    if final_win_rate > 0.5 {
        println!("✅ SUCCESS: Genetic AI is now performing well!");

        // Update the main library constants
        update_library_constants(&best_params);
        println!("🔧 Library constants updated!");
    } else {
        println!("❌ ISSUE: Genetic AI still needs improvement");
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
}
