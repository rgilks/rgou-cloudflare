use rand::prelude::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use rgou_ai_core::{GameState, Player, AI};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufReader, Write};
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize, Debug)]
struct Config {
    matchups: Vec<Matchup>,
}

#[derive(Deserialize, Debug)]
struct Matchup {
    name: String,
    player1: PlayerConfig,
    player2: PlayerConfig,
    games: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct PlayerConfig {
    #[serde(rename = "type")]
    kind: String,
    depth: Option<u8>,
}

// Remove ML1, ML2, MLTest from AIType, use ML(String) instead
enum AIType {
    EMM(u8),
    Random,
    ML(String),
}

// Load ML model mapping from JSON
fn load_ml_model_map() -> HashMap<String, String> {
    let path = "ml_models.json";
    let file = File::open(path).expect("Failed to open ml_models.json");
    let json: JsonValue = serde_json::from_reader(file).expect("Failed to parse ml_models.json");
    let mut map = HashMap::new();
    for (k, v) in json.as_object().unwrap() {
        map.insert(k.to_lowercase(), v.as_str().unwrap().to_string());
    }
    map
}

fn get_ai_type(cfg: &PlayerConfig) -> AIType {
    match cfg.kind.to_lowercase().as_str() {
        "expectiminimax" | "emm" => AIType::EMM(cfg.depth.unwrap_or(1)),
        "random" => AIType::Random,
        s if s.starts_with("ml") => AIType::ML(s.to_string()),
        _ => panic!("Unknown AI type: {}", cfg.kind),
    }
}

struct GameStats {
    winner: Player,
    p1_time_s: f64,
    p2_time_s: f64,
    p1_moves: usize,
    p2_moves: usize,
}

fn play_game(p1: &AIType, p2: &AIType, ml_model_map: &HashMap<String, String>) -> GameStats {
    let mut game_state = GameState::new();
    let mut ai1 = AI::new();
    let mut ai2 = AI::new();
    let mut rng = rand::thread_rng();
    let mut p1_time_s = 0.0f64;
    let mut p2_time_s = 0.0f64;
    let mut p1_moves = 0usize;
    let mut p2_moves = 0usize;
    let mut ml1 = None;
    let mut ml2 = None;
    let mut ml1_loaded = false;
    let mut ml2_loaded = false;
    loop {
        let current_player = game_state.current_player;
        game_state.dice_roll = rng.gen_range(0..=4);
        if game_state.dice_roll == 0 {
            game_state.current_player = game_state.current_player.opponent();
            continue;
        }
        if current_player == Player::Player1 {
            let start = Instant::now();
            let mv = match p1 {
                AIType::EMM(depth) => ai1.get_best_move(&game_state, *depth).0,
                AIType::Random => {
                    let moves = game_state.get_valid_moves();
                    if moves.is_empty() {
                        None
                    } else {
                        Some(*moves.choose(&mut rng).unwrap())
                    }
                }
                AIType::ML(model) => {
                    if ml1.is_none() {
                        let weights_path = ml_model_map
                            .get(&model.to_lowercase())
                            .expect(&format!("ML model '{}' not found in ml_models.json", model));
                        let (v, p) = load_ml_weights_generic(weights_path);
                        ml1 = Some(new_ml_ai(&v, &p));
                        ml1_loaded = true;
                    }
                    ml1.as_mut().unwrap().get_best_move(&game_state).r#move
                }
            };
            p1_time_s += start.elapsed().as_secs_f64();
            p1_moves += 1;
            if let Some(piece_index) = mv {
                game_state.make_move(piece_index).unwrap();
            } else {
                game_state.current_player = game_state.current_player.opponent();
            }
        } else {
            let start = Instant::now();
            let mv = match p2 {
                AIType::EMM(depth) => ai2.get_best_move(&game_state, *depth).0,
                AIType::Random => {
                    let moves = game_state.get_valid_moves();
                    if moves.is_empty() {
                        None
                    } else {
                        Some(*moves.choose(&mut rng).unwrap())
                    }
                }
                AIType::ML(model) => {
                    if ml2.is_none() {
                        let weights_path = ml_model_map
                            .get(&model.to_lowercase())
                            .expect(&format!("ML model '{}' not found in ml_models.json", model));
                        let (v, p) = load_ml_weights_generic(weights_path);
                        ml2 = Some(new_ml_ai(&v, &p));
                        ml2_loaded = true;
                    }
                    ml2.as_mut().unwrap().get_best_move(&game_state).r#move
                }
            };
            p2_time_s += start.elapsed().as_secs_f64();
            p2_moves += 1;
            if let Some(piece_index) = mv {
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
                p1_time_s,
                p2_time_s,
                p1_moves,
                p2_moves,
            };
        }
    }
}

// --- ML AI helpers ---
use rgou_ai_core::ml_ai::MLAI;
use serde_json;

fn load_ml_weights_generic(weights_path: &str) -> (Vec<f32>, Vec<f32>) {
    let path = Path::new(weights_path);
    let content = std::fs::read_to_string(&path)
        .expect(&format!("Failed to read weights file: {}", path.display()));
    let weights: serde_json::Value = serde_json::from_str(&content).unwrap();
    let value_weights = weights["valueWeights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let policy_weights = weights["policyWeights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    (value_weights, policy_weights)
}
fn new_ml_ai(value: &[f32], policy: &[f32]) -> MLAI {
    let mut ml = MLAI::new();
    ml.load_pretrained(value, policy);
    ml
}

fn main() {
    let config_path = "ai_benchmark_config.json";
    let file = File::open(config_path).expect("Failed to open config file");
    let reader = BufReader::new(file);
    let config: Config = serde_json::from_reader(reader).expect("Failed to parse config");
    let ml_model_map = load_ml_model_map();
    println!("matchup\tplayer1\tplayer2\twin_rate_p1\twin_rate_p2\tavg_time_ms_p1\tavg_time_ms_p2");

    // Open CSV file for writing (overwrite)
    let mut csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ai_benchmark_results.csv")
        .expect("Failed to open ai_benchmark_results.csv for writing");
    writeln!(
        csv_file,
        "matchup,player1,player2,win_rate_p1,win_rate_p2,avg_time_ms_p1,avg_time_ms_p2"
    )
    .unwrap();

    for matchup in config.matchups {
        let p1_type = get_ai_type(&matchup.player1);
        let p2_type = get_ai_type(&matchup.player2);
        let results: Vec<_> = (0..matchup.games)
            .into_par_iter()
            .map(|_| play_game(&p1_type, &p2_type, &ml_model_map))
            .collect();
        let p1_wins = results
            .iter()
            .filter(|s| s.winner == Player::Player1)
            .count();
        let p2_wins = results
            .iter()
            .filter(|s| s.winner == Player::Player2)
            .count();
        let total_p1_time: f64 = results.iter().map(|s| s.p1_time_s).sum();
        let total_p2_time: f64 = results.iter().map(|s| s.p2_time_s).sum();
        let total_p1_moves: usize = results.iter().map(|s| s.p1_moves).sum();
        let total_p2_moves: usize = results.iter().map(|s| s.p2_moves).sum();
        let win_rate_p1 = p1_wins as f64 / matchup.games as f64;
        let win_rate_p2 = p2_wins as f64 / matchup.games as f64;
        let avg_time_ms_p1 = if total_p1_moves > 0 {
            (total_p1_time / total_p1_moves as f64) * 1000.0
        } else {
            0.0
        };
        let avg_time_ms_p2 = if total_p2_moves > 0 {
            (total_p2_time / total_p2_moves as f64) * 1000.0
        } else {
            0.0
        };
        let p1_label = match &p1_type {
            AIType::EMM(d) => format!("EMM-{}", d),
            AIType::Random => "Random".to_string(),
            AIType::ML(model) => format!("ML-{}", model),
        };
        let p2_label = match &p2_type {
            AIType::EMM(d) => format!("EMM-{}", d),
            AIType::Random => "Random".to_string(),
            AIType::ML(model) => format!("ML-{}", model),
        };
        println!(
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.4}\t{:.4}",
            matchup.name,
            p1_label,
            p2_label,
            win_rate_p1,
            win_rate_p2,
            avg_time_ms_p1,
            avg_time_ms_p2
        );
        writeln!(
            csv_file,
            "{},{},{},{:.3},{:.3},{:.4},{:.4}",
            matchup.name,
            p1_label,
            p2_label,
            win_rate_p1,
            win_rate_p2,
            avg_time_ms_p1,
            avg_time_ms_p2
        )
        .unwrap();
    }
}
