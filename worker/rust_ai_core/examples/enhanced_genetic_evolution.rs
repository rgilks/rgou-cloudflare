use rgou_ai_core::genetic_ai::{GeneticAlgorithm, HeuristicParams};
use std::time::Instant;

fn main() {
    println!("🧬 ENHANCED GENETIC ALGORITHM EVOLUTION");
    println!("=======================================");
    println!("Optimized for better parameter fine-tuning");
    println!();

    // Enhanced configuration for better optimization
    let ga = GeneticAlgorithm::new(
        50,   // population_size (increased from 20)
        0.05, // mutation_rate (decreased for finer tuning)
        5,    // tournament_size (increased for better selection)
        50,   // games_per_individual (increased from 10)
    );

    println!("Enhanced Configuration:");
    println!("  Population size: {}", 50);
    println!("  Mutation rate: {:.1}%", 0.05 * 100.0);
    println!("  Tournament size: {}", 5);
    println!("  Games per individual: {}", 50);
    println!("  Generations: {}", 100);
    println!();

    let start_time = Instant::now();
    let best_params = ga.evolve(100);
    let total_time = start_time.elapsed();

    println!("\n🎯 EVOLUTION COMPLETE!");
    println!(
        "Total evolution time: {:.2} seconds",
        total_time.as_secs_f64()
    );
    println!();

    println!("Best evolved parameters:");
    println!("  win_score: {}", best_params.win_score);
    println!(
        "  finished_piece_value: {}",
        best_params.finished_piece_value
    );
    println!("  position_weight: {}", best_params.position_weight);
    println!("  advancement_bonus: {}", best_params.advancement_bonus);
    println!("  rosette_safety_bonus: {}", best_params.rosette_safety_bonus);
    println!("  rosette_chain_bonus: {}", best_params.rosette_chain_bonus);
    println!("  capture_bonus: {}", best_params.capture_bonus);
    println!("  vulnerability_penalty: {}", best_params.vulnerability_penalty);
    println!("  center_control_bonus: {}", best_params.center_control_bonus);
    println!("  piece_coordination_bonus: {}", best_params.piece_coordination_bonus);
    println!("  blocking_bonus: {}", best_params.blocking_bonus);
    println!("  early_game_bonus: {}", best_params.early_game_bonus);
    println!("  late_game_urgency: {}", best_params.late_game_urgency);
    println!("  turn_order_bonus: {}", best_params.turn_order_bonus);
    println!("  mobility_bonus: {}", best_params.mobility_bonus);
    println!("  attack_pressure_bonus: {}", best_params.attack_pressure_bonus);
    println!("  defensive_structure_bonus: {}", best_params.defensive_structure_bonus);

    let default_params = HeuristicParams::new();
    println!("\n📊 COMPARISON WITH DEFAULT PARAMETERS:");
    println!(
        "  win_score: {} → {} (diff: {})",
        default_params.win_score,
        best_params.win_score,
        best_params.win_score - default_params.win_score
    );
    println!(
        "  finished_piece_value: {} → {} (diff: {})",
        default_params.finished_piece_value,
        best_params.finished_piece_value,
        best_params.finished_piece_value - default_params.finished_piece_value
    );
    println!(
        "  position_weight: {} → {} (diff: {})",
        default_params.position_weight,
        best_params.position_weight,
        best_params.position_weight - default_params.position_weight
    );
    println!(
        "  rosette_safety_bonus: {} → {} (diff: {})",
        default_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus,
        best_params.rosette_safety_bonus - default_params.rosette_safety_bonus
    );
    println!(
        "  rosette_chain_bonus: {} → {} (diff: {})",
        default_params.rosette_chain_bonus,
        best_params.rosette_chain_bonus,
        best_params.rosette_chain_bonus - default_params.rosette_chain_bonus
    );
    println!(
        "  advancement_bonus: {} → {} (diff: {})",
        default_params.advancement_bonus,
        best_params.advancement_bonus,
        best_params.advancement_bonus - default_params.advancement_bonus
    );
    println!(
        "  capture_bonus: {} → {} (diff: {})",
        default_params.capture_bonus,
        best_params.capture_bonus,
        best_params.capture_bonus - default_params.capture_bonus
    );
    println!(
        "  center_control_bonus: {} → {} (diff: {})",
        default_params.center_control_bonus,
        best_params.center_control_bonus,
        best_params.center_control_bonus - default_params.center_control_bonus
    );

    // Calculate percentage changes
    println!("\n📈 PERCENTAGE CHANGES:");
    println!(
        "  win_score: {:.1}%",
        ((best_params.win_score as f64 / default_params.win_score as f64) - 1.0) * 100.0
    );
    println!(
        "  finished_piece_value: {:.1}%",
        ((best_params.finished_piece_value as f64 / default_params.finished_piece_value as f64)
            - 1.0)
            * 100.0
    );
    println!(
        "  position_weight: {:.1}%",
        ((best_params.position_weight as f64 / default_params.position_weight as f64) - 1.0)
            * 100.0
    );
    println!(
        "  rosette_safety_bonus: {:.1}%",
        ((best_params.rosette_safety_bonus as f64 / default_params.rosette_safety_bonus as f64) - 1.0) * 100.0
    );
    println!(
        "  rosette_chain_bonus: {:.1}%",
        ((best_params.rosette_chain_bonus as f64 / default_params.rosette_chain_bonus as f64)
            - 1.0)
            * 100.0
    );
    println!(
        "  advancement_bonus: {:.1}%",
        ((best_params.advancement_bonus as f64 / default_params.advancement_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  capture_bonus: {:.1}%",
        ((best_params.capture_bonus as f64 / default_params.capture_bonus as f64) - 1.0) * 100.0
    );
    println!(
        "  center_control_bonus: {:.1}%",
        ((best_params.center_control_bonus as f64 / default_params.center_control_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  piece_coordination_bonus: {:.1}%",
        ((best_params.piece_coordination_bonus as f64 / default_params.piece_coordination_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  vulnerability_penalty: {:.1}%",
        ((best_params.vulnerability_penalty as f64 / default_params.vulnerability_penalty as f64) - 1.0)
            * 100.0
    );
    println!(
        "  blocking_bonus: {:.1}%",
        ((best_params.blocking_bonus as f64 / default_params.blocking_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  early_game_bonus: {:.1}%",
        ((best_params.early_game_bonus as f64 / default_params.early_game_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  rosette_chain_bonus: {:.1}%",
        ((best_params.rosette_chain_bonus as f64 / default_params.rosette_chain_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  late_game_urgency: {:.1}%",
        ((best_params.late_game_urgency as f64 / default_params.late_game_urgency as f64) - 1.0)
            * 100.0
    );
    println!(
        "  turn_order_bonus: {:.1}%",
        ((best_params.turn_order_bonus as f64 / default_params.turn_order_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  mobility_bonus: {:.1}%",
        ((best_params.mobility_bonus as f64 / default_params.mobility_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  attack_pressure_bonus: {:.1}%",
        ((best_params.attack_pressure_bonus as f64 / default_params.attack_pressure_bonus as f64) - 1.0)
            * 100.0
    );
    println!(
        "  defensive_structure_bonus: {:.1}%",
        ((best_params.defensive_structure_bonus as f64 / default_params.defensive_structure_bonus as f64) - 1.0)
            * 100.0
    );

    // Save parameters to file for later use
    let params_json = serde_json::to_string_pretty(&best_params).unwrap();
    std::fs::write("evolved_params.json", params_json).expect("Failed to write parameters");
    println!("\n💾 Parameters saved to 'evolved_params.json'");

    // Generate Rust code for easy integration
    println!("\n🔧 RUST CODE FOR INTEGRATION:");
    println!("```rust");
    println!("let evolved_params = HeuristicParams {{");
    println!("    win_score: {},", best_params.win_score);
    println!(
        "    finished_piece_value: {},",
        best_params.finished_piece_value
    );
    println!("    position_weight: {},", best_params.position_weight);
    println!("    advancement_bonus: {},", best_params.advancement_bonus);
    println!("    rosette_safety_bonus: {},", best_params.rosette_safety_bonus);
    println!("    rosette_chain_bonus: {},", best_params.rosette_chain_bonus);
    println!("    capture_bonus: {},", best_params.capture_bonus);
    println!("    vulnerability_penalty: {},", best_params.vulnerability_penalty);
    println!("    center_control_bonus: {},", best_params.center_control_bonus);
    println!("    piece_coordination_bonus: {},", best_params.piece_coordination_bonus);
    println!("    blocking_bonus: {},", best_params.blocking_bonus);
    println!("    early_game_bonus: {},", best_params.early_game_bonus);
    println!("    late_game_urgency: {},", best_params.late_game_urgency);
    println!("    turn_order_bonus: {},", best_params.turn_order_bonus);
    println!("    mobility_bonus: {},", best_params.mobility_bonus);
    println!("    attack_pressure_bonus: {},", best_params.attack_pressure_bonus);
    println!("    defensive_structure_bonus: {},", best_params.defensive_structure_bonus);
    println!("}};");
    println!("```");
}
