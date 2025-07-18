use rand::Rng;
use rayon::prelude::*;
use rgou_ai_core::{GameState, Player, AI};
use std::io::{self, Write};
use std::time::Instant;

struct GameStats {
    winner: Player,
    ai1_time_s: f64,
    ai1_moves: usize,
}

fn play_game(depth: u8) -> GameStats {
    let mut game_state = GameState::new();
    let mut ai1 = AI::new();
    let mut ai2 = AI::new();
    let mut rng = rand::thread_rng();
    let mut ai1_time_s = 0.0f64;
    let mut ai1_moves = 0usize;
    loop {
        let current_player = game_state.current_player;
        game_state.dice_roll = rng.gen_range(0..=4);
        if game_state.dice_roll == 0 {
            game_state.current_player = game_state.current_player.opponent();
            continue;
        }
        if current_player == Player::Player1 {
            let start = Instant::now();
            let (best_move, _) = ai1.get_best_move(&game_state, depth);
            ai1_time_s += start.elapsed().as_secs_f64();
            ai1_moves += 1;
            if let Some(piece_index) = best_move {
                game_state.make_move(piece_index).unwrap();
            } else {
                game_state.current_player = game_state.current_player.opponent();
            }
        } else {
            let (best_move, _) = ai2.get_best_move(&game_state, 1);
            if let Some(piece_index) = best_move {
                game_state.make_move(piece_index).unwrap();
            } else {
                game_state.current_player = game_state.current_player.opponent();
            }
        }
        if game_state.is_game_over() {
            let p1_finished = game_state
                .player1_pieces
                .iter()
                .filter(|p| p.square == 20)
                .count();
            let winner = if p1_finished == rgou_ai_core::PIECES_PER_PLAYER {
                Player::Player1
            } else {
                Player::Player2
            };
            return GameStats {
                winner,
                ai1_time_s,
                ai1_moves,
            };
        }
    }
}

fn main() {
    let num_games = 100;
    println!("depth,win_rate,ai1_wins,ai2_wins,avg_time_ms");
    for depth in 1..=7 {
        let results: Vec<_> = (0..num_games)
            .into_par_iter()
            .map(|_| {
                let stats = play_game(depth);
                print!(".");
                io::stdout().flush().unwrap();
                stats
            })
            .collect();
        println!("");
        let ai1_wins = results
            .iter()
            .filter(|s| s.winner == Player::Player1)
            .count();
        let ai2_wins = results
            .iter()
            .filter(|s| s.winner == Player::Player2)
            .count();
        let total_time_s: f64 = results.iter().map(|s| s.ai1_time_s).sum();
        let total_moves: usize = results.iter().map(|s| s.ai1_moves).sum();
        let win_rate = ai1_wins as f64 / num_games as f64;
        let avg_time_ms = if total_moves > 0 {
            (total_time_s / total_moves as f64) * 1000.0
        } else {
            0.0
        };
        println!(
            "{},{:.3},{},{},{:.4}",
            depth, win_rate, ai1_wins, ai2_wins, avg_time_ms
        );
    }
}
