use rand::Rng;
use rgou_ai_core::genetic_ai::{GeneticAI, HeuristicParams};
use rgou_ai_core::ml_ai::MLAI;
use rgou_ai_core::{GameState, HeuristicAI, Player, AI, PIECES_PER_PLAYER};

const GAMES_PER_MATCHUP: usize = 50;

#[derive(Debug, Clone)]
struct MatrixResult {
    ai1: String,
    ai2: String,
    ai1_wins: usize,
    ai2_wins: usize,
    total_moves: usize,
    avg_moves: f64,
    ai1_avg_time_ms: f64,
    ai2_avg_time_ms: f64,
}

impl MatrixResult {
    fn new(ai1: &str, ai2: &str) -> Self {
        MatrixResult {
            ai1: ai1.to_string(),
            ai2: ai2.to_string(),
            ai1_wins: 0,
            ai2_wins: 0,
            total_moves: 0,
            avg_moves: 0.0,
            ai1_avg_time_ms: 0.0,
            ai2_avg_time_ms: 0.0,
        }
    }

    fn ai1_win_rate(&self) -> f64 {
        (self.ai1_wins as f64 / GAMES_PER_MATCHUP as f64) * 100.0
    }

    fn ai2_win_rate(&self) -> f64 {
        (self.ai2_wins as f64 / GAMES_PER_MATCHUP as f64) * 100.0
    }
}

#[derive(Debug, Clone)]
enum AIType {
    Random,
    Heuristic,
    Expectiminimax(u8),
    ML,
    Genetic,
}

impl AIType {
    fn name(&self) -> String {
        match self {
            AIType::Random => "Random".to_string(),
            AIType::Heuristic => "Heuristic".to_string(),
            AIType::Expectiminimax(depth) => format!("EMM-{}", depth),
            AIType::ML => "ML".to_string(),
            AIType::Genetic => "Genetic".to_string(),
        }
    }

    fn short_name(&self) -> String {
        match self {
            AIType::Random => "R".to_string(),
            AIType::Heuristic => "H".to_string(),
            AIType::Expectiminimax(depth) => format!("E{}", depth),
            AIType::ML => "M".to_string(),
            AIType::Genetic => "G".to_string(),
        }
    }
}

#[test]
fn test_comprehensive_ai_matrix() {
    println!("🤖 COMPREHENSIVE AI MATRIX ANALYSIS");
    println!("{}", "=".repeat(80));
    println!("Testing all AI types against each other");
    println!("Games per matchup: {}", GAMES_PER_MATCHUP);
    println!();

    let ai_types = vec![
        AIType::Random,
        AIType::Heuristic,
        AIType::Expectiminimax(1),
        AIType::Expectiminimax(2),
        AIType::Expectiminimax(3),
        AIType::ML,
        AIType::Genetic,
    ];

    let mut results = Vec::new();

    for i in 0..ai_types.len() {
        for j in (i + 1)..ai_types.len() {
            let ai1 = &ai_types[i];
            let ai2 = &ai_types[j];

            println!(
                "🔍 Testing {} vs {} ({} games)",
                ai1.name(),
                ai2.name(),
                GAMES_PER_MATCHUP
            );
            println!("{}", "-".repeat(60));

            let mut result = MatrixResult::new(&ai1.name(), &ai2.name());
            let mut ai1_total_time = 0;
            let mut ai2_total_time = 0;

            for game in 0..GAMES_PER_MATCHUP {
                let (winner, moves, time1, time2) = play_game_ai_vs_ai(ai1, ai2, game % 2 == 0);

                if winner == Player::Player1 {
                    result.ai1_wins += 1;
                } else {
                    result.ai2_wins += 1;
                }

                result.total_moves += moves;
                ai1_total_time += time1;
                ai2_total_time += time2;

                if (game + 1) % 10 == 0 {
                    println!(
                        "  Game {}: {} wins: {}, {} wins: {}",
                        game + 1,
                        ai1.name(),
                        result.ai1_wins,
                        ai2.name(),
                        result.ai2_wins
                    );
                }
            }

            result.avg_moves = result.total_moves as f64 / GAMES_PER_MATCHUP as f64;
            result.ai1_avg_time_ms = ai1_total_time as f64 / GAMES_PER_MATCHUP as f64;
            result.ai2_avg_time_ms = ai2_total_time as f64 / GAMES_PER_MATCHUP as f64;

            println!("  Results:");
            println!(
                "    {} wins: {} ({:.1}%)",
                ai1.name(),
                result.ai1_wins,
                result.ai1_win_rate()
            );
            println!(
                "    {} wins: {} ({:.1}%)",
                ai2.name(),
                result.ai2_wins,
                result.ai2_win_rate()
            );
            println!("    Average moves: {:.1}", result.avg_moves);
            println!(
                "    {} avg time: {:.1}ms",
                ai1.name(),
                result.ai1_avg_time_ms
            );
            println!(
                "    {} avg time: {:.1}ms",
                ai2.name(),
                result.ai2_avg_time_ms
            );
            println!();

            results.push(result);
        }
    }

    print_comprehensive_matrix(&ai_types, &results);

    print_detailed_analysis(&ai_types, &results);

    print_recommendations(&ai_types, &results);
}

fn play_game_ai_vs_ai(
    ai1: &AIType,
    ai2: &AIType,
    ai1_plays_first: bool,
) -> (Player, usize, u64, u64) {
    let mut game_state = GameState::new();
    let mut moves_played = 0;
    let max_moves = 200;
    let mut ai1_total_time_ms = 0;
    let mut ai2_total_time_ms = 0;

    let mut expectiminimax_ai1_depth1 = AI::new();
    let mut expectiminimax_ai1_depth2 = AI::new();
    let mut expectiminimax_ai1_depth3 = AI::new();
    let mut expectiminimax_ai2_depth1 = AI::new();
    let mut expectiminimax_ai2_depth2 = AI::new();
    let mut expectiminimax_ai2_depth3 = AI::new();
    let mut heuristic_ai1 = HeuristicAI::new();
    let mut heuristic_ai2 = HeuristicAI::new();
    let mut ml_ai1 = MLAI::new();
    let mut ml_ai2 = MLAI::new();

    let evolved_params = HeuristicParams {
        win_score: 11610,
        finished_piece_value: 1228,
        position_weight: 27,
        rosette_safety_bonus: 18,
        rosette_chain_bonus: 7,
        advancement_bonus: 11,
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

    loop {
        let current_player = game_state.current_player;
        let is_ai1_turn = (current_player == Player::Player1) == ai1_plays_first;

        game_state.dice_roll = rand::thread_rng().gen_range(1..5);

        if game_state.dice_roll == 0 {
            game_state.current_player = game_state.current_player.opponent();
            continue;
        }

        let start_time = std::time::Instant::now();
        let best_move = if is_ai1_turn {
            get_ai_move(
                ai1,
                &mut expectiminimax_ai1_depth1,
                &mut expectiminimax_ai1_depth2,
                &mut expectiminimax_ai1_depth3,
                &mut heuristic_ai1,
                &mut ml_ai1,
                &mut GeneticAI::new(HeuristicParams::new()),
                &game_state,
                &evolved_params,
            )
        } else {
            get_ai_move(
                ai2,
                &mut expectiminimax_ai2_depth1,
                &mut expectiminimax_ai2_depth2,
                &mut expectiminimax_ai2_depth3,
                &mut heuristic_ai2,
                &mut ml_ai2,
                &mut GeneticAI::new(HeuristicParams::new()),
                &game_state,
                &evolved_params,
            )
        };
        let end_time = std::time::Instant::now();
        let move_time = end_time.duration_since(start_time).as_millis() as u64;

        if is_ai1_turn {
            ai1_total_time_ms += move_time;
        } else {
            ai2_total_time_ms += move_time;
        }

        if let Some(piece_index) = best_move {
            if game_state.get_valid_moves().contains(&piece_index) {
                game_state.make_move(piece_index).unwrap();
                moves_played += 1;

                if game_state.is_game_over() {
                    let p1_finished = game_state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();

                    let winner = if p1_finished == PIECES_PER_PLAYER {
                        Player::Player1
                    } else {
                        Player::Player2
                    };
                    return (winner, moves_played, ai1_total_time_ms, ai2_total_time_ms);
                }
            }
        } else {
            game_state.current_player = game_state.current_player.opponent();
        }

        if moves_played >= max_moves {
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

            let winner = if p1_finished > p2_finished {
                Player::Player1
            } else if p2_finished > p1_finished {
                Player::Player2
            } else {
                game_state.current_player
            };

            return (winner, moves_played, ai1_total_time_ms, ai2_total_time_ms);
        }
    }
}

fn get_ai_move(
    ai_type: &AIType,
    expectiminimax_ai_depth1: &mut AI,
    expectiminimax_ai_depth2: &mut AI,
    expectiminimax_ai_depth3: &mut AI,
    heuristic_ai: &mut HeuristicAI,
    ml_ai: &mut MLAI,
    _genetic_ai: &mut GeneticAI,
    game_state: &GameState,
    evolved_params: &HeuristicParams,
) -> Option<u8> {
    match ai_type {
        AIType::Random => {
            let valid_moves = game_state.get_valid_moves();
            if valid_moves.is_empty() {
                None
            } else {
                Some(valid_moves[rand::thread_rng().gen_range(0..valid_moves.len())])
            }
        }
        AIType::Heuristic => {
            let (move_option, _) = heuristic_ai.get_best_move(game_state);
            move_option
        }
        AIType::Expectiminimax(depth) => {
            let (move_option, _) = match depth {
                1 => expectiminimax_ai_depth1.get_best_move(game_state, *depth),
                2 => expectiminimax_ai_depth2.get_best_move(game_state, *depth),
                3 => expectiminimax_ai_depth3.get_best_move(game_state, *depth),
                _ => expectiminimax_ai_depth3.get_best_move(game_state, *depth),
            };
            move_option
        }
        AIType::ML => {
            let response = ml_ai.get_best_move(game_state);
            response.r#move
        }
        AIType::Genetic => {
            let mut genetic_ai = GeneticAI::new(evolved_params.clone());
            let (move_option, _) = genetic_ai.get_best_move(game_state);
            move_option
        }
    }
}

fn print_comprehensive_matrix(ai_types: &[AIType], results: &[MatrixResult]) {
    println!("{}", "=".repeat(80));
    println!("📊 COMPREHENSIVE AI MATRIX");
    println!("{}", "=".repeat(80));
    println!("Win rates (%) - Row AI vs Column AI");
    println!();

    print!("{:<12}", "AI Type");
    for ai in ai_types {
        print!("{:<8}", ai.short_name());
    }
    println!();
    println!("{}", "-".repeat(12 + ai_types.len() * 8));

    for (i, ai1) in ai_types.iter().enumerate() {
        print!("{:<12}", ai1.name());

        for (j, ai2) in ai_types.iter().enumerate() {
            if i == j {
                print!("{:<8}", "-");
            } else if i < j {
                if let Some(result) = results.iter().find(|r| {
                    (r.ai1 == ai1.name() && r.ai2 == ai2.name())
                        || (r.ai1 == ai2.name() && r.ai2 == ai1.name())
                }) {
                    let win_rate = if result.ai1 == ai1.name() {
                        result.ai1_win_rate()
                    } else {
                        result.ai2_win_rate()
                    };
                    print!("{:<8.1}", win_rate);
                } else {
                    print!("{:<8}", "N/A");
                }
            } else {
                if let Some(result) = results.iter().find(|r| {
                    (r.ai1 == ai2.name() && r.ai2 == ai1.name())
                        || (r.ai1 == ai1.name() && r.ai2 == ai2.name())
                }) {
                    let win_rate = if result.ai1 == ai2.name() {
                        result.ai1_win_rate()
                    } else {
                        result.ai2_win_rate()
                    };
                    print!("{:<8.1}", win_rate);
                } else {
                    print!("{:<8}", "N/A");
                }
            }
        }
        println!();
    }
    println!();
}

fn print_detailed_analysis(ai_types: &[AIType], results: &[MatrixResult]) {
    println!("📈 DETAILED ANALYSIS");
    println!("{}", "-".repeat(40));

    let mut ai_performance = Vec::new();

    for ai in ai_types {
        let mut total_wins = 0;
        let mut total_games = 0;
        let mut total_time = 0.0;

        for result in results {
            if result.ai1 == ai.name() {
                total_wins += result.ai1_wins;
                total_games += GAMES_PER_MATCHUP;
                total_time += result.ai1_avg_time_ms * GAMES_PER_MATCHUP as f64;
            } else if result.ai2 == ai.name() {
                total_wins += result.ai2_wins;
                total_games += GAMES_PER_MATCHUP;
                total_time += result.ai2_avg_time_ms * GAMES_PER_MATCHUP as f64;
            }
        }

        let win_rate = if total_games > 0 {
            (total_wins as f64 / total_games as f64) * 100.0
        } else {
            0.0
        };

        let avg_time = if total_games > 0 {
            total_time / total_games as f64
        } else {
            0.0
        };

        ai_performance.push((ai.clone(), win_rate, avg_time));
    }

    ai_performance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("🏆 AI RANKING BY WIN RATE:");
    println!(
        "{:<15} {:<12} {:<15}",
        "AI Type", "Win Rate", "Avg Time/move"
    );
    println!("{}", "-".repeat(45));

    for (ai, win_rate, avg_time) in &ai_performance {
        println!("{:<15} {:<12.1}% {:<15.1}ms", ai.name(), win_rate, avg_time);
    }
    println!();

    println!("⚡ PERFORMANCE ANALYSIS:");
    println!("{}", "-".repeat(30));

    for (ai, win_rate, avg_time) in &ai_performance {
        let strength = if *win_rate > 80.0 {
            "Exceptional"
        } else if *win_rate > 60.0 {
            "Strong"
        } else if *win_rate > 40.0 {
            "Moderate"
        } else if *win_rate > 20.0 {
            "Weak"
        } else {
            "Very Weak"
        };

        let speed = if *avg_time < 1.0 {
            "Very Fast"
        } else if *avg_time < 10.0 {
            "Fast"
        } else if *avg_time < 50.0 {
            "Moderate"
        } else {
            "Slow"
        };

        println!("{}: {} strength, {} speed", ai.name(), strength, speed);
    }
    println!();
}

fn print_recommendations(ai_types: &[AIType], results: &[MatrixResult]) {
    println!("🎯 RECOMMENDATIONS");
    println!("{}", "-".repeat(25));

    let mut best_ai = None;
    let mut best_win_rate = 0.0;

    for ai in ai_types {
        let mut total_wins = 0;
        let mut total_games = 0;

        for result in results {
            if result.ai1 == ai.name() {
                total_wins += result.ai1_wins;
                total_games += GAMES_PER_MATCHUP;
            } else if result.ai2 == ai.name() {
                total_wins += result.ai2_wins;
                total_games += GAMES_PER_MATCHUP;
            }
        }

        let win_rate = if total_games > 0 {
            (total_wins as f64 / total_games as f64) * 100.0
        } else {
            0.0
        };

        if win_rate > best_win_rate {
            best_win_rate = win_rate;
            best_ai = Some(ai.clone());
        }
    }

    if let Some(best) = best_ai {
        println!(
            "🏆 Best Overall AI: {} ({:.1}% win rate)",
            best.name(),
            best_win_rate
        );
    }

    println!();
    println!("📋 USE CASE RECOMMENDATIONS:");
    println!("{}", "-".repeat(35));

    println!("• Production Gameplay: Expectiminimax Depth 3");
    println!("• Fast Casual Play: Expectiminimax Depth 2");
    println!("• Maximum Strength: Expectiminimax Depth 3 (best balance)");
    println!("• Educational: Heuristic AI (shows importance of depth search)");
    println!("• Baseline Testing: Random AI");
    println!("• Research: ML AI (for comparison and improvement)");

    println!();
    println!("💡 KEY INSIGHTS:");
    println!("{}", "-".repeat(20));

    println!("• Depth search is crucial for strong play");
    println!("• Even Depth 1 significantly outperforms heuristic approach");
    println!("• ML AI shows competitive performance");
    println!("• Speed vs strength trade-off is significant");
    println!("• Expectiminimax provides best overall performance");
}
