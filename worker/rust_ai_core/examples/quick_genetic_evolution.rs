use rgou_ai_core::genetic_ai::{GeneticAlgorithm, HeuristicParams};
use std::time::Instant;

fn main() {
    println!("🚀 QUICK GENETIC ALGORITHM EVOLUTION (15 min target)");
    println!("==================================================");

    let start_time = Instant::now();

    let ga = GeneticAlgorithm::new(
        20,   // population_size (smaller for speed)
        0.15, // mutation_rate (higher for faster exploration)
        3,    // tournament_size
        20,   // games_per_individual (fewer games for speed)
    );

    println!("Quick Configuration:");
    println!("  Population size: {}", 20);
    println!("  Mutation rate: {:.1}%", 0.15 * 100.0);
    println!("  Tournament size: {}", 3);
    println!("  Games per individual: {}", 20);
    println!("  Target time: ~15 minutes");
    println!();

    let best_params = ga.evolve(30); // 30 generations should take ~15 minutes

    let elapsed = start_time.elapsed();
    println!(
        "\n⏱️  Evolution Complete in {:.1} minutes!",
        elapsed.as_secs_f64() / 60.0
    );

    println!("\n🏆 Best evolved parameters:");
    println!("  win_score: {}", best_params.win_score);
    println!(
        "  finished_piece_value: {}",
        best_params.finished_piece_value
    );
    println!("  position_weight: {}", best_params.position_weight);
    println!(
        "  rosette_safety_bonus: {}",
        best_params.rosette_safety_bonus
    );
    println!("  rosette_chain_bonus: {}", best_params.rosette_chain_bonus);
    println!("  advancement_bonus: {}", best_params.advancement_bonus);
    println!("  capture_bonus: {}", best_params.capture_bonus);
    println!(
        "  vulnerability_penalty: {}",
        best_params.vulnerability_penalty
    );
    println!(
        "  center_control_bonus: {}",
        best_params.center_control_bonus
    );
    println!(
        "  piece_coordination_bonus: {}",
        best_params.piece_coordination_bonus
    );
    println!("  blocking_bonus: {}", best_params.blocking_bonus);
    println!("  early_game_bonus: {}", best_params.early_game_bonus);
    println!("  late_game_urgency: {}", best_params.late_game_urgency);
    println!("  turn_order_bonus: {}", best_params.turn_order_bonus);
    println!("  mobility_bonus: {}", best_params.mobility_bonus);
    println!(
        "  attack_pressure_bonus: {}",
        best_params.attack_pressure_bonus
    );
    println!(
        "  defensive_structure_bonus: {}",
        best_params.defensive_structure_bonus
    );

    let default_params = HeuristicParams::new();
    println!("\n📊 Comparison with default parameters:");
    println!(
        "  win_score: {} → {} (diff: {:+})",
        default_params.win_score,
        best_params.win_score,
        best_params.win_score - default_params.win_score
    );
    println!(
        "  finished_piece_value: {} → {} (diff: {:+})",
        default_params.finished_piece_value,
        best_params.finished_piece_value,
        best_params.finished_piece_value - default_params.finished_piece_value
    );
    println!(
        "  position_weight: {} → {} (diff: {:+})",
        default_params.position_weight,
        best_params.position_weight,
        best_params.position_weight - default_params.position_weight
    );
    println!(
        "  rosette_safety_bonus: {} → {} (diff: {:+})",
        default_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus - default_params.rosette_safety_bonus
    );
    println!(
        "  capture_bonus: {} → {} (diff: {:+})",
        default_params.capture_bonus,
        best_params.capture_bonus,
        best_params.capture_bonus - default_params.capture_bonus
    );
    println!(
        "  center_control_bonus: {} → {} (diff: {:+})",
        default_params.center_control_bonus,
        best_params.center_control_bonus,
        best_params.center_control_bonus - default_params.center_control_bonus
    );

    println!("\n🎯 Ready for model comparison!");
    println!("Run: cargo run --example ai_matrix_analysis");
}
