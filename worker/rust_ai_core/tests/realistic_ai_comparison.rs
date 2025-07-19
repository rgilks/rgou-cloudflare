use rand::Rng;
use rgou_ai_core::{
    genetic_ai::{GeneticAI, HeuristicParams},
    GameState, HeuristicAI, Player, AI, roll_tetrahedral_dice,
};

const REALISTIC_GAMES: usize = 20;

#[test]
fn test_realistic_ai_comparison() {
    println!("🎯 REALISTIC AI COMPARISON");
    println!("{}", "=".repeat(60));
    println!("Testing AI performance with realistic expectations");
    println!("Games per matchup: {}", REALISTIC_GAMES);
    println!();

    println!("🔍 Test 1: EMM-2 vs Heuristic AI");
    test_emm2_vs_heuristic();
    println!();

    println!("🔍 Test 2: EMM-3 vs Heuristic AI");
    test_emm3_vs_heuristic();
    println!();

    println!("🧬 Test 3: Genetic AI vs Heuristic AI");
    test_genetic_vs_heuristic();
    println!();

    println!("🎲 Test 4: Random vs Heuristic AI");
    test_random_vs_heuristic();
    println!();

    println!("✅ Realistic comparison complete!");
}

fn test_emm2_vs_heuristic() {
    let mut emm_ai = AI::new();
    let mut heuristic_ai = HeuristicAI::new();

    let mut emm_wins = 0;
    let mut heuristic_wins = 0;

    for game in 0..REALISTIC_GAMES {
        let (winner, _) = play_single_game_emm(&mut emm_ai, &mut heuristic_ai, game % 2 == 0);

        if winner == Player::Player1 {
            emm_wins += 1;
        } else {
            heuristic_wins += 1;
        }

        if (game + 1) % 50 == 0 {
            println!(
                "  Game {}: EMM-2 {} wins, Heuristic {} wins",
                game + 1,
                emm_wins,
                heuristic_wins
            );
        }
    }

    let emm_rate = (emm_wins as f64 / REALISTIC_GAMES as f64) * 100.0;
    println!(
        "  Final: EMM-2 {} wins ({:.1}%), Heuristic {} wins ({:.1}%)",
        emm_wins,
        emm_rate,
        heuristic_wins,
        100.0 - emm_rate
    );


    if emm_rate > 55.0 {
        println!("  ✅ EMM-2 performs as expected (stronger than heuristic)");
    } else {
        println!("  ❌ EMM-2 underperforming (should be stronger than heuristic)");
    }
}

fn test_emm3_vs_heuristic() {
    let mut emm_ai = AI::new();
    let mut heuristic_ai = HeuristicAI::new();

    let mut emm_wins = 0;
    let mut heuristic_wins = 0;

    for game in 0..REALISTIC_GAMES {
        let (winner, _) =
            play_single_game_emm_depth3(&mut emm_ai, &mut heuristic_ai, game % 2 == 0);

        if winner == Player::Player1 {
            emm_wins += 1;
        } else {
            heuristic_wins += 1;
        }

        if (game + 1) % 50 == 0 {
            println!(
                "  Game {}: EMM-3 {} wins, Heuristic {} wins",
                game + 1,
                emm_wins,
                heuristic_wins
            );
        }
    }

    let emm_rate = (emm_wins as f64 / REALISTIC_GAMES as f64) * 100.0;
    println!(
        "  Final: EMM-3 {} wins ({:.1}%), Heuristic {} wins ({:.1}%)",
        emm_wins,
        emm_rate,
        heuristic_wins,
        100.0 - emm_rate
    );


    if emm_rate > 55.0 {
        println!("  ✅ EMM-3 performs as expected (stronger than heuristic)");
    } else {
        println!("  ❌ EMM-3 underperforming (should be stronger than heuristic)");
    }
}

fn test_genetic_vs_heuristic() {
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

    let mut genetic_ai = GeneticAI::new(evolved_params);
    let mut heuristic_ai = HeuristicAI::new();

    let mut genetic_wins = 0;
    let mut heuristic_wins = 0;

    for game in 0..REALISTIC_GAMES {
        let (winner, _) = play_single_game(&mut genetic_ai, &mut heuristic_ai, game % 2 == 0);

        if winner == Player::Player1 {
            genetic_wins += 1;
        } else {
            heuristic_wins += 1;
        }

        if (game + 1) % 50 == 0 {
            println!(
                "  Game {}: Genetic {} wins, Heuristic {} wins",
                game + 1,
                genetic_wins,
                heuristic_wins
            );
        }
    }

    let genetic_rate = (genetic_wins as f64 / REALISTIC_GAMES as f64) * 100.0;
    println!(
        "  Final: Genetic {} wins ({:.1}%), Heuristic {} wins ({:.1}%)",
        genetic_wins,
        genetic_rate,
        heuristic_wins,
        100.0 - genetic_rate
    );


    if genetic_rate > 48.0 && genetic_rate < 65.0 {
        println!("  ✅ Genetic AI performs as expected (close to heuristic)");
    } else {
        println!("  ❌ Genetic AI performance unexpected");
    }
}

fn test_random_vs_heuristic() {
    let mut heuristic_ai = HeuristicAI::new();

    let mut random_wins = 0;
    let mut heuristic_wins = 0;

    for game in 0..REALISTIC_GAMES {
        let (winner, _) = play_single_game_random(&mut heuristic_ai, game % 2 == 0);

        if winner == Player::Player1 {
            random_wins += 1;
        } else {
            heuristic_wins += 1;
        }

        if (game + 1) % 50 == 0 {
            println!(
                "  Game {}: Random {} wins, Heuristic {} wins",
                game + 1,
                random_wins,
                heuristic_wins
            );
        }
    }

    let random_rate = (random_wins as f64 / REALISTIC_GAMES as f64) * 100.0;
    println!(
        "  Final: Random {} wins ({:.1}%), Heuristic {} wins ({:.1}%)",
        random_wins,
        random_rate,
        heuristic_wins,
        100.0 - random_rate
    );


    if random_rate < 50.0 {
        println!("  ✅ Heuristic AI performs as expected (beats random)");
    } else {
        println!("  ❌ Heuristic AI underperforming (should beat random)");
    }
}

fn play_single_game(
    ai1: &mut GeneticAI,
    ai2: &mut HeuristicAI,
    ai1_plays_first: bool,
) -> (Player, usize) {
    let mut state = GameState::new();
    let mut moves = 0;
    let max_moves = 200;

    loop {
        let current_player = state.current_player;
        let is_ai1_turn = (current_player == Player::Player1) == ai1_plays_first;

        state.dice_roll = roll_tetrahedral_dice();

        if state.dice_roll == 0 {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let best_move = if is_ai1_turn {
            let (move_option, _) = ai1.get_best_move(&state);
            move_option
        } else {
            let (move_option, _) = ai2.get_best_move(&state);
            move_option
        };

        if let Some(piece_index) = best_move {
            if state.get_valid_moves().contains(&piece_index) {
                state.make_move(piece_index).unwrap();
                moves += 1;

                if state.is_game_over() {
                    let p1_finished = state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    return if p1_finished == 7 {
                        (Player::Player1, moves)
                    } else {
                        (Player::Player2, moves)
                    };
                }
            }
        } else {
            state.current_player = state.current_player.opponent();
        }

        if moves >= max_moves {
            let p1_finished = state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let p2_finished = state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            return if p1_finished > p2_finished {
                (Player::Player1, moves)
            } else {
                (Player::Player2, moves)
            };
        }
    }
}

fn play_single_game_emm(
    ai1: &mut AI,
    ai2: &mut HeuristicAI,
    ai1_plays_first: bool,
) -> (Player, usize) {
    let mut state = GameState::new();
    let mut moves = 0;
    let max_moves = 200;

    loop {
        let current_player = state.current_player;
        let is_ai1_turn = (current_player == Player::Player1) == ai1_plays_first;

        state.dice_roll = roll_tetrahedral_dice();

        if state.dice_roll == 0 {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let best_move = if is_ai1_turn {
            let (move_option, _) = ai1.get_best_move(&state, 2);
            move_option
        } else {
            let (move_option, _) = ai2.get_best_move(&state);
            move_option
        };

        if let Some(piece_index) = best_move {
            if state.get_valid_moves().contains(&piece_index) {
                state.make_move(piece_index).unwrap();
                moves += 1;

                if state.is_game_over() {
                    let p1_finished = state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    return if p1_finished == 7 {
                        (Player::Player1, moves)
                    } else {
                        (Player::Player2, moves)
                    };
                }
            }
        } else {
            state.current_player = state.current_player.opponent();
        }

        if moves >= max_moves {
            let p1_finished = state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let p2_finished = state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            return if p1_finished > p2_finished {
                (Player::Player1, moves)
            } else {
                (Player::Player2, moves)
            };
        }
    }
}

fn play_single_game_emm_depth3(
    ai1: &mut AI,
    ai2: &mut HeuristicAI,
    ai1_plays_first: bool,
) -> (Player, usize) {
    let mut state = GameState::new();
    let mut moves = 0;
    let max_moves = 200;

    loop {
        let current_player = state.current_player;
        let is_ai1_turn = (current_player == Player::Player1) == ai1_plays_first;

        state.dice_roll = roll_tetrahedral_dice();

        if state.dice_roll == 0 {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let best_move = if is_ai1_turn {
            let (move_option, _) = ai1.get_best_move(&state, 3);
            move_option
        } else {
            let (move_option, _) = ai2.get_best_move(&state);
            move_option
        };

        if let Some(piece_index) = best_move {
            if state.get_valid_moves().contains(&piece_index) {
                state.make_move(piece_index).unwrap();
                moves += 1;

                if state.is_game_over() {
                    let p1_finished = state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    return if p1_finished == 7 {
                        (Player::Player1, moves)
                    } else {
                        (Player::Player2, moves)
                    };
                }
            }
        } else {
            state.current_player = state.current_player.opponent();
        }

        if moves >= max_moves {
            let p1_finished = state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let p2_finished = state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            return if p1_finished > p2_finished {
                (Player::Player1, moves)
            } else {
                (Player::Player2, moves)
            };
        }
    }
}

fn play_single_game_random(ai2: &mut HeuristicAI, random_plays_first: bool) -> (Player, usize) {
    let mut state = GameState::new();
    let mut moves = 0;
    let max_moves = 200;

    loop {
        let current_player = state.current_player;
        let is_random_turn = (current_player == Player::Player1) == random_plays_first;

        state.dice_roll = rand::thread_rng().gen_range(1..5);

        if state.dice_roll == 0 {
            state.current_player = state.current_player.opponent();
            continue;
        }

        let best_move = if is_random_turn {
            let valid_moves = state.get_valid_moves();
            if valid_moves.is_empty() {
                None
            } else {
                Some(valid_moves[rand::thread_rng().gen_range(0..valid_moves.len())])
            }
        } else {
            let (move_option, _) = ai2.get_best_move(&state);
            move_option
        };

        if let Some(piece_index) = best_move {
            if state.get_valid_moves().contains(&piece_index) {
                state.make_move(piece_index).unwrap();
                moves += 1;

                if state.is_game_over() {
                    let p1_finished = state
                        .player1_pieces
                        .iter()
                        .filter(|p| p.square == 20)
                        .count();
                    return if p1_finished == 7 {
                        (Player::Player1, moves)
                    } else {
                        (Player::Player2, moves)
                    };
                }
            }
        } else {
            state.current_player = state.current_player.opponent();
        }

        if moves >= max_moves {
            let p1_finished = state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let p2_finished = state
                .player2_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            return if p1_finished > p2_finished {
                (Player::Player1, moves)
            } else {
                (Player::Player2, moves)
            };
        }
    }
}
