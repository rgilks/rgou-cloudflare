use rgou_ai_core::genetic_ai::{GeneticAlgorithm, HeuristicParams};

fn main() {
    println!("Starting Genetic Algorithm Evolution for Royal Game of Ur AI");
    println!("==========================================================");

    let ga = GeneticAlgorithm::new(
        20,  // population_size
        0.1, // mutation_rate
        3,   // tournament_size
        10,  // games_per_individual
    );

    println!("Configuration:");
    println!("  Population size: {}", 20);
    println!("  Mutation rate: {:.1}%", 0.1 * 100.0);
    println!("  Tournament size: {}", 3);
    println!("  Games per individual: {}", 10);
    println!("  Generations: {}", 50);
    println!();

    let best_params = ga.evolve(50);

    println!("\nEvolution Complete!");
    println!("Best evolved parameters:");
    println!("  win_score: {}", best_params.win_score);
    println!(
        "  finished_piece_value: {}",
        best_params.finished_piece_value
    );
    println!("  position_weight: {}", best_params.position_weight);
    println!("  safety_bonus: {}", best_params.safety_bonus);
    println!(
        "  rosette_control_bonus: {}",
        best_params.rosette_control_bonus
    );
    println!("  advancement_bonus: {}", best_params.advancement_bonus);
    println!("  capture_bonus: {}", best_params.capture_bonus);
    println!("  center_lane_bonus: {}", best_params.center_lane_bonus);

    let default_params = HeuristicParams::new();
    println!("\nComparison with default parameters:");
    println!(
        "  win_score: {} -> {} (diff: {})",
        default_params.win_score,
        best_params.win_score,
        best_params.win_score - default_params.win_score
    );
    println!(
        "  finished_piece_value: {} -> {} (diff: {})",
        default_params.finished_piece_value,
        best_params.finished_piece_value,
        best_params.finished_piece_value - default_params.finished_piece_value
    );
    println!(
        "  position_weight: {} -> {} (diff: {})",
        default_params.position_weight,
        best_params.position_weight,
        best_params.position_weight - default_params.position_weight
    );
    println!(
        "  safety_bonus: {} -> {} (diff: {})",
        default_params.safety_bonus,
        best_params.safety_bonus,
        best_params.safety_bonus - default_params.safety_bonus
    );
    println!(
        "  rosette_control_bonus: {} -> {} (diff: {})",
        default_params.rosette_control_bonus,
        best_params.rosette_control_bonus,
        best_params.rosette_control_bonus - default_params.rosette_control_bonus
    );
    println!(
        "  advancement_bonus: {} -> {} (diff: {})",
        default_params.advancement_bonus,
        best_params.advancement_bonus,
        best_params.advancement_bonus - default_params.advancement_bonus
    );
    println!(
        "  capture_bonus: {} -> {} (diff: {})",
        default_params.capture_bonus,
        best_params.capture_bonus,
        best_params.capture_bonus - default_params.capture_bonus
    );
    println!(
        "  center_lane_bonus: {} -> {} (diff: {})",
        default_params.center_lane_bonus,
        best_params.center_lane_bonus,
        best_params.center_lane_bonus - default_params.center_lane_bonus
    );
}
