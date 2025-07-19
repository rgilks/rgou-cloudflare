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
        "  rosette_safety_bonus: {} -> {} (diff: {})",
        default_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus - default_params.rosette_safety_bonus
    );
    println!(
        "  rosette_chain_bonus: {} -> {} (diff: {})",
        default_params.rosette_chain_bonus,
        best_params.rosette_chain_bonus,
        best_params.rosette_chain_bonus - default_params.rosette_chain_bonus
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
        "  vulnerability_penalty: {} -> {} (diff: {})",
        default_params.vulnerability_penalty,
        best_params.vulnerability_penalty,
        best_params.vulnerability_penalty - default_params.vulnerability_penalty
    );
    println!(
        "  center_control_bonus: {} -> {} (diff: {})",
        default_params.center_control_bonus,
        best_params.center_control_bonus,
        best_params.center_control_bonus - default_params.center_control_bonus
    );
    println!(
        "  piece_coordination_bonus: {} -> {} (diff: {})",
        default_params.piece_coordination_bonus,
        best_params.piece_coordination_bonus,
        best_params.piece_coordination_bonus - default_params.piece_coordination_bonus
    );
    println!(
        "  blocking_bonus: {} -> {} (diff: {})",
        default_params.blocking_bonus,
        best_params.blocking_bonus,
        best_params.blocking_bonus - default_params.blocking_bonus
    );
    println!(
        "  early_game_bonus: {} -> {} (diff: {})",
        default_params.early_game_bonus,
        best_params.early_game_bonus,
        best_params.early_game_bonus - default_params.early_game_bonus
    );
    println!(
        "  late_game_urgency: {} -> {} (diff: {})",
        default_params.late_game_urgency,
        best_params.late_game_urgency,
        best_params.late_game_urgency - default_params.late_game_urgency
    );
    println!(
        "  turn_order_bonus: {} -> {} (diff: {})",
        default_params.turn_order_bonus,
        best_params.turn_order_bonus,
        best_params.turn_order_bonus - default_params.turn_order_bonus
    );
    println!(
        "  mobility_bonus: {} -> {} (diff: {})",
        default_params.mobility_bonus,
        best_params.mobility_bonus,
        best_params.mobility_bonus - default_params.mobility_bonus
    );
    println!(
        "  attack_pressure_bonus: {} -> {} (diff: {})",
        default_params.attack_pressure_bonus,
        best_params.attack_pressure_bonus,
        best_params.attack_pressure_bonus - default_params.attack_pressure_bonus
    );
    println!(
        "  defensive_structure_bonus: {} -> {} (diff: {})",
        default_params.defensive_structure_bonus,
        best_params.defensive_structure_bonus,
        best_params.defensive_structure_bonus - default_params.defensive_structure_bonus
    );
}
