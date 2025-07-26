//! Genetic parameter evolution for EMM and Heuristic AIs

use rand::seq::SliceRandom;
use rayon::prelude::*;
use rgou_ai_core::Player;
use rgou_ai_core::{genetic_params::GeneticParams, GameState, AI};
use std::fs;

const POPULATION_SIZE: usize = 12;
const GENERATIONS: usize = 8;
const GAMES_PER_EVAL: usize = 20;
const MUTATION_RATE: f64 = 0.3;
const MUTATION_STRENGTH: f64 = 1.0;
const CROSSOVER_RATE: f64 = 0.5;

fn optimize_cpu_usage() {
    if cfg!(target_os = "macos") {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let optimal_threads = (num_cores as f64 * 0.8) as usize;
        rayon::ThreadPoolBuilder::new()
            .num_threads(optimal_threads)
            .stack_size(8 * 1024 * 1024)
            .build_global()
            .unwrap_or_else(|_| {
                println!("Warning: Could not set optimal thread count, using default");
            });
        println!(
            "🍎 Apple Silicon detected: Using {} threads ({} cores available)",
            optimal_threads, num_cores
        );
    } else {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cores)
            .stack_size(8 * 1024 * 1024)
            .build_global()
            .unwrap_or_else(|_| {
                println!("Warning: Could not set optimal thread count, using default");
            });
        println!("🖥️  Using {} threads for parallel processing", num_cores);
    }
}

// Tournament-style evaluation: evolved params vs default params
fn evaluate_params_tournament(evolved_params: &GeneticParams) -> f64 {
    let default_params = GeneticParams::default();

    let results: Vec<bool> = (0..GAMES_PER_EVAL)
        .into_par_iter()
        .map(|_| {
            let mut game_state = GameState::new();
            let mut moves_played = 0;
            let max_moves = 200;

            while !game_state.is_game_over() && moves_played < max_moves {
                let current_player = game_state.current_player;
                let is_evolved_turn = current_player == Player::Player2;

                game_state.dice_roll = rgou_ai_core::dice::roll_dice();
                if game_state.dice_roll == 0 {
                    game_state.current_player = game_state.current_player.opponent();
                    continue;
                }

                // Use different parameters based on whose turn it is
                let test_params = if is_evolved_turn {
                    evolved_params.clone()
                } else {
                    default_params.clone()
                };

                // Create a new game state with the test parameters
                let mut test_state = GameState::with_genetic_params(test_params);
                test_state.board = game_state.board.clone();
                test_state.player1_pieces = game_state.player1_pieces.clone();
                test_state.player2_pieces = game_state.player2_pieces.clone();
                test_state.current_player = game_state.current_player;
                test_state.dice_roll = game_state.dice_roll;

                let mut ai = AI::new();
                let (best_move, _) = ai.get_best_move(&test_state, 3);

                if let Some(move_piece) = best_move {
                    game_state.make_move(move_piece).ok();
                } else {
                    game_state.current_player = game_state.current_player.opponent();
                }
                moves_played += 1;
            }

            // Determine winner - evolved params are Player2
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

            if p2_finished >= 7 {
                true // Evolved params win
            } else if p1_finished >= 7 {
                false // Default params win
            } else {
                // Game ended by move limit, evaluate final position
                let evolved_eval = game_state.evaluate();
                evolved_eval > 0 // Positive eval means Player2 (evolved) is winning
            }
        })
        .collect();

    let wins = results.iter().filter(|&&won| won).count();
    wins as f64 / GAMES_PER_EVAL as f64
}

fn main() {
    println!("\n=== Genetic Parameter Evolution for EMM AI (Tournament Style) ===");
    println!("🚀 Optimizing CPU usage for maximum performance...");
    optimize_cpu_usage();

    let start_time = std::time::Instant::now();
    let mut rng = rand::thread_rng();
    let mut population: Vec<GeneticParams> = (0..POPULATION_SIZE)
        .map(|_| GeneticParams::default().random_mutation(1.0, 2.0))
        .collect();
    let mut best_score = 0.0;
    let mut best_params = GeneticParams::default();

    for gen in 0..GENERATIONS {
        println!("\n🧬 Generation {}", gen + 1);
        let gen_start = std::time::Instant::now();

        // Tournament evaluation against default parameters
        let scored: Vec<(f64, GeneticParams)> = population
            .par_iter()
            .map(|p| {
                let score = evaluate_params_tournament(p);
                (score, p.clone())
            })
            .collect();

        let mut scored = scored;
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let (top_score, top_params) = &scored[0];
        let gen_duration = gen_start.elapsed();

        println!("  ⏱️  Generation time: {:.2}s", gen_duration.as_secs_f64());
        println!("  🏆 Best score vs defaults: {:.2}", top_score);
        println!(
            "  📊 Average score: {:.2}",
            scored.iter().map(|(s, _)| s).sum::<f64>() / scored.len() as f64
        );

        if *top_score > best_score {
            best_score = *top_score;
            best_params = top_params.clone();
            println!("  🎉 New best score achieved!");
        }

        // Elitism: keep top 2
        let mut new_population = vec![scored[0].1.clone(), scored[1].1.clone()];

        // Fill rest with crossover/mutation
        while new_population.len() < POPULATION_SIZE {
            let parent1 = scored.choose(&mut rng).unwrap().1.clone();
            let parent2 = scored.choose(&mut rng).unwrap().1.clone();
            let mut child = parent1.crossover(&parent2, CROSSOVER_RATE);
            child = child.random_mutation(MUTATION_RATE, MUTATION_STRENGTH);
            new_population.push(child);
        }
        population = new_population;
    }

    let total_duration = start_time.elapsed();

    println!("\n🎉 === Evolution Complete ===");
    println!("⏱️  Total time: {:.2}s", total_duration.as_secs_f64());
    println!("🏆 Best win rate vs defaults: {:.2}", best_score);
    println!("🔧 Best parameters: {:#?}", best_params);

    // Save to file
    let out_path = "../../ml/data/genetic_params/evolved.json";
    fs::write(
        out_path,
        serde_json::to_string_pretty(&best_params).unwrap(),
    )
    .unwrap();
    println!("💾 Saved best parameters to {}", out_path);
}
