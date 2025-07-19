use rgou_ai_core::genetic_ai::{GeneticAI, GeneticAlgorithm, HeuristicParams};
use rgou_ai_core::{roll_tetrahedral_dice, GameState, HeuristicAI, Player};
use std::time::{Duration, Instant};

fn main() {
    println!("🔍 GENETIC AI DIAGNOSTIC");
    println!("=======================");
    println!("Quick diagnostic to identify evaluation issues");
    println!("Target duration: 3 minutes");
    println!();

    let start_time = Instant::now();
    let target_duration = Duration::from_secs(3 * 60); // 3 minutes

    // Test 1: Compare default vs current best parameters
    println!("📊 TEST 1: Default vs Current Best Parameters");
    println!("=============================================");

    let default_params = HeuristicParams::new();
    let current_best =
        HeuristicParams::from_file("best_genetic_params.json").unwrap_or(default_params.clone());

    let mut default_ai = GeneticAI::new(default_params.clone());
    let mut best_ai = GeneticAI::new(current_best.clone());
    let mut heuristic_ai = HeuristicAI::new();

    // Test default vs heuristic
    let (default_wins, default_total) = play_multiple_games(&mut default_ai, &mut heuristic_ai, 20);
    let default_win_rate = default_wins as f64 / default_total as f64;

    // Test best vs heuristic
    let (best_wins, best_total) = play_multiple_games(&mut best_ai, &mut heuristic_ai, 20);
    let best_win_rate = best_wins as f64 / best_total as f64;

    println!(
        "Default params vs Heuristic: {}/{} = {:.1}%",
        default_wins,
        default_total,
        default_win_rate * 100.0
    );
    println!(
        "Best params vs Heuristic: {}/{} = {:.1}%",
        best_wins,
        best_total,
        best_win_rate * 100.0
    );
    println!();

    // Test 2: Quick genetic evolution with better evaluation
    println!("🧬 TEST 2: Quick Genetic Evolution");
    println!("=================================");

    let ga = GeneticAlgorithm::new(
        15,  // Smaller population
        0.1, // Higher mutation rate for exploration
        3,   // Smaller tournament
        15,  // Fewer games per individual
    );

    let mut generation = 0;
    let mut best_params = default_params.clone();
    let mut best_fitness = default_win_rate * 100.0;

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
        if fitness > 60.0 {
            println!("\n🎯 Good performance achieved!");
            break;
        }

        // Stop if we're not improving
        if generation > 10 && best_fitness < 45.0 {
            println!("\n⚠️  Poor performance detected - stopping early");
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "\n✅ Diagnostic completed in {:.2} minutes",
        total_time.as_secs_f64() / 60.0
    );
    println!("🏆 Best fitness achieved: {:.1}%", best_fitness);
    println!("📊 Total generations: {}", generation);

    // Save results
    let json_params = serde_json::to_string_pretty(&best_params).unwrap();
    std::fs::write("diagnostic_best_params.json", json_params).unwrap();
    println!("💾 Best parameters saved to: diagnostic_best_params.json");

    // Test 3: Verify the best parameters
    println!("\n🔍 TEST 3: Verification");
    println!("=======================");

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
        println!("✅ SUCCESS: Genetic AI is performing well!");
    } else {
        println!("❌ ISSUE: Genetic AI performance is poor - needs investigation");
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
