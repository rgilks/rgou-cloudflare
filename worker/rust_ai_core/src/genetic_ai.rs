use crate::{GameState, Player, PIECES_PER_PLAYER};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::thread;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeuristicParams {
    // Core game objectives
    pub win_score: i32,
    pub finished_piece_value: i32,

    // Position evaluation
    pub position_weight: i32,
    pub advancement_bonus: i32,

    // Safety and control
    pub rosette_safety_bonus: i32, // Combined safety + control
    pub rosette_chain_bonus: i32,

    // Tactical elements
    pub capture_bonus: i32,
    pub vulnerability_penalty: i32,

    // Strategic positioning
    pub center_control_bonus: i32, // Combined center lane + dominance
    pub piece_coordination_bonus: i32,
    pub blocking_bonus: i32,

    // Game phase awareness
    pub early_game_bonus: i32, // New: early development
    pub late_game_urgency: i32,
    pub turn_order_bonus: i32,

    // Advanced tactics
    pub mobility_bonus: i32,            // New: move availability
    pub attack_pressure_bonus: i32,     // New: threatening opponent
    pub defensive_structure_bonus: i32, // New: overall defense
}

impl HeuristicParams {
    pub fn new() -> Self {
        Self {
            win_score: 10000,
            finished_piece_value: 1000,
            position_weight: 15,
            advancement_bonus: 5,
            rosette_safety_bonus: 25,
            rosette_chain_bonus: 12,
            capture_bonus: 35,
            vulnerability_penalty: 20,
            center_control_bonus: 8,
            piece_coordination_bonus: 10,
            blocking_bonus: 15,
            early_game_bonus: 10,
            late_game_urgency: 30,
            turn_order_bonus: 5,
            mobility_bonus: 10,
            attack_pressure_bonus: 10,
            defensive_structure_bonus: 10,
        }
    }

    pub fn from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(filename)?;
        let params: HeuristicParams = serde_json::from_str(&content)?;
        Ok(params)
    }

    pub fn random() -> Self {
        let mut rng = thread_rng();
        Self {
            win_score: rng.gen_range(5000..15000),
            finished_piece_value: rng.gen_range(500..2000),
            position_weight: rng.gen_range(5..30),
            advancement_bonus: rng.gen_range(1..15),
            rosette_safety_bonus: rng.gen_range(10..50),
            rosette_chain_bonus: rng.gen_range(5..25),
            capture_bonus: rng.gen_range(20..60),
            vulnerability_penalty: rng.gen_range(10..40),
            center_control_bonus: rng.gen_range(3..20),
            piece_coordination_bonus: rng.gen_range(5..25),
            blocking_bonus: rng.gen_range(5..30),
            early_game_bonus: rng.gen_range(5..20),
            late_game_urgency: rng.gen_range(15..60),
            turn_order_bonus: rng.gen_range(1..15),
            mobility_bonus: rng.gen_range(5..20),
            attack_pressure_bonus: rng.gen_range(5..20),
            defensive_structure_bonus: rng.gen_range(5..20),
        }
    }

    pub fn mutate(&self, mutation_rate: f64) -> Self {
        let mut rng = thread_rng();
        let mut new_params = self.clone();

        // Enhanced mutation with adaptive ranges and parameter-specific tuning
        if rng.gen_bool(mutation_rate) {
            // Win score: larger range for significant changes
            new_params.win_score = (new_params.win_score as f64 * rng.gen_range(0.7..1.3)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Finished piece value: moderate range
            new_params.finished_piece_value =
                (new_params.finished_piece_value as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Position weight: fine-tuning range
            new_params.position_weight =
                (new_params.position_weight as f64 * rng.gen_range(0.9..1.1)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Advancement bonus: fine-tuning range
            new_params.advancement_bonus =
                (new_params.advancement_bonus as f64 * rng.gen_range(0.9..1.1)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Rosette safety bonus: moderate range
            new_params.rosette_safety_bonus =
                (new_params.rosette_safety_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Rosette chain bonus: moderate range
            new_params.rosette_chain_bonus =
                (new_params.rosette_chain_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Capture bonus: moderate range
            new_params.capture_bonus =
                (new_params.capture_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Vulnerability penalty: moderate range
            new_params.vulnerability_penalty =
                (new_params.vulnerability_penalty as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Center control bonus: moderate range
            new_params.center_control_bonus =
                (new_params.center_control_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Piece coordination bonus: moderate range
            new_params.piece_coordination_bonus =
                (new_params.piece_coordination_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Blocking bonus: moderate range
            new_params.blocking_bonus =
                (new_params.blocking_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Early game bonus: moderate range
            new_params.early_game_bonus =
                (new_params.early_game_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Late game urgency: larger range for significant changes
            new_params.late_game_urgency =
                (new_params.late_game_urgency as f64 * rng.gen_range(0.7..1.3)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Turn order bonus: fine-tuning range
            new_params.turn_order_bonus =
                (new_params.turn_order_bonus as f64 * rng.gen_range(0.9..1.1)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Mobility bonus: moderate range
            new_params.mobility_bonus =
                (new_params.mobility_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Attack pressure bonus: moderate range
            new_params.attack_pressure_bonus =
                (new_params.attack_pressure_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            // Defensive structure bonus: moderate range
            new_params.defensive_structure_bonus =
                (new_params.defensive_structure_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }

        // Ensure parameters stay within reasonable bounds
        new_params.win_score = new_params.win_score.max(1000).min(20000);
        new_params.finished_piece_value = new_params.finished_piece_value.max(100).min(3000);
        new_params.position_weight = new_params.position_weight.max(1).min(100);
        new_params.advancement_bonus = new_params.advancement_bonus.max(1).min(50);
        new_params.rosette_safety_bonus = new_params.rosette_safety_bonus.max(5).min(100);
        new_params.rosette_chain_bonus = new_params.rosette_chain_bonus.max(1).min(50);
        new_params.capture_bonus = new_params.capture_bonus.max(10).min(100);
        new_params.vulnerability_penalty = new_params.vulnerability_penalty.max(5).min(80);
        new_params.center_control_bonus = new_params.center_control_bonus.max(1).min(40);
        new_params.piece_coordination_bonus = new_params.piece_coordination_bonus.max(1).min(50);
        new_params.blocking_bonus = new_params.blocking_bonus.max(1).min(60);
        new_params.early_game_bonus = new_params.early_game_bonus.max(1).min(30);
        new_params.late_game_urgency = new_params.late_game_urgency.max(5).min(100);
        new_params.turn_order_bonus = new_params.turn_order_bonus.max(1).min(30);
        new_params.mobility_bonus = new_params.mobility_bonus.max(1).min(30);
        new_params.attack_pressure_bonus = new_params.attack_pressure_bonus.max(1).min(30);
        new_params.defensive_structure_bonus = new_params.defensive_structure_bonus.max(1).min(30);

        new_params
    }

    pub fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        Self {
            win_score: if rng.gen_bool(0.5) {
                self.win_score
            } else {
                other.win_score
            },
            finished_piece_value: if rng.gen_bool(0.5) {
                self.finished_piece_value
            } else {
                other.finished_piece_value
            },
            position_weight: if rng.gen_bool(0.5) {
                self.position_weight
            } else {
                other.position_weight
            },
            advancement_bonus: if rng.gen_bool(0.5) {
                self.advancement_bonus
            } else {
                other.advancement_bonus
            },
            rosette_safety_bonus: if rng.gen_bool(0.5) {
                self.rosette_safety_bonus
            } else {
                other.rosette_safety_bonus
            },
            rosette_chain_bonus: if rng.gen_bool(0.5) {
                self.rosette_chain_bonus
            } else {
                other.rosette_chain_bonus
            },
            capture_bonus: if rng.gen_bool(0.5) {
                self.capture_bonus
            } else {
                other.capture_bonus
            },
            vulnerability_penalty: if rng.gen_bool(0.5) {
                self.vulnerability_penalty
            } else {
                other.vulnerability_penalty
            },
            center_control_bonus: if rng.gen_bool(0.5) {
                self.center_control_bonus
            } else {
                other.center_control_bonus
            },
            piece_coordination_bonus: if rng.gen_bool(0.5) {
                self.piece_coordination_bonus
            } else {
                other.piece_coordination_bonus
            },
            blocking_bonus: if rng.gen_bool(0.5) {
                self.blocking_bonus
            } else {
                other.blocking_bonus
            },
            early_game_bonus: if rng.gen_bool(0.5) {
                self.early_game_bonus
            } else {
                other.early_game_bonus
            },
            late_game_urgency: if rng.gen_bool(0.5) {
                self.late_game_urgency
            } else {
                other.late_game_urgency
            },
            turn_order_bonus: if rng.gen_bool(0.5) {
                self.turn_order_bonus
            } else {
                other.turn_order_bonus
            },
            mobility_bonus: if rng.gen_bool(0.5) {
                self.mobility_bonus
            } else {
                other.mobility_bonus
            },
            attack_pressure_bonus: if rng.gen_bool(0.5) {
                self.attack_pressure_bonus
            } else {
                other.attack_pressure_bonus
            },
            defensive_structure_bonus: if rng.gen_bool(0.5) {
                self.defensive_structure_bonus
            } else {
                other.defensive_structure_bonus
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneticIndividual {
    pub params: HeuristicParams,
    pub fitness: f64,
    pub games_played: u32,
    pub wins: u32,
}

impl GeneticIndividual {
    pub fn new(params: HeuristicParams) -> Self {
        Self {
            params,
            fitness: 0.0,
            games_played: 0,
            wins: 0,
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.games_played == 0 {
            0.0
        } else {
            self.wins as f64 / self.games_played as f64
        }
    }
}

pub struct GeneticAI {
    pub nodes_evaluated: u32,
    params: HeuristicParams,
}

impl GeneticAI {
    pub fn new(params: HeuristicParams) -> Self {
        Self {
            nodes_evaluated: 0,
            params,
        }
    }

    pub fn get_best_move(&mut self, state: &GameState) -> (Option<u8>, Vec<crate::MoveEvaluation>) {
        self.nodes_evaluated = 0;
        let valid_moves = state.get_valid_moves();

        if valid_moves.is_empty() {
            return (None, vec![]);
        }

        let is_maximizing = state.current_player == Player::Player2;
        let mut best_move = None;
        let mut best_score = if is_maximizing { f32::MIN } else { f32::MAX };
        let mut move_evaluations = Vec::new();

        for &piece_index in &valid_moves {
            let mut test_state = state.clone();
            if let Ok(()) = test_state.make_move(piece_index) {
                let score = self.evaluate_with_params(&test_state, &self.params) as f32;
                self.nodes_evaluated += 1;

                let from_square =
                    state.get_pieces(state.current_player)[piece_index as usize].square;
                let to_square = if test_state.get_pieces(state.current_player)[piece_index as usize]
                    .square
                    == 20
                {
                    None
                } else {
                    Some(
                        test_state.get_pieces(state.current_player)[piece_index as usize].square
                            as u8,
                    )
                };

                let move_type = if from_square == -1 {
                    "move".to_string()
                } else if to_square.is_some()
                    && test_state.board[to_square.unwrap() as usize].is_some()
                {
                    "capture".to_string()
                } else {
                    "move".to_string()
                };

                move_evaluations.push(crate::MoveEvaluation {
                    piece_index,
                    score,
                    move_type,
                    from_square,
                    to_square,
                });

                if is_maximizing {
                    if score > best_score {
                        best_score = score;
                        best_move = Some(piece_index);
                    }
                } else {
                    if score < best_score {
                        best_score = score;
                        best_move = Some(piece_index);
                    }
                }
            }
        }

        move_evaluations.sort_by(|a, b| {
            if is_maximizing {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        (best_move, move_evaluations)
    }

    fn evaluate_with_params(&self, state: &GameState, params: &HeuristicParams) -> i32 {
        let mut p1_finished = 0;
        let mut p2_finished = 0;
        let mut p1_on_board = 0;
        let mut p2_on_board = 0;

        for piece in &state.player1_pieces {
            if piece.square == 20 {
                p1_finished += 1;
            } else if piece.square > -1 {
                p1_on_board += 1;
            }
        }

        for piece in &state.player2_pieces {
            if piece.square == 20 {
                p2_finished += 1;
            } else if piece.square > -1 {
                p2_on_board += 1;
            }
        }

        if p1_finished == PIECES_PER_PLAYER as i32 {
            return -params.win_score;
        }
        if p2_finished == PIECES_PER_PLAYER as i32 {
            return params.win_score;
        }

        let mut score = (p2_finished - p1_finished) * params.finished_piece_value;
        score += (p2_on_board - p1_on_board) * params.capture_bonus;

        let (p1_pos_score, p1_strategic_score) =
            self.evaluate_player_position_with_params(state, Player::Player1, params);
        let (p2_pos_score, p2_strategic_score) =
            self.evaluate_player_position_with_params(state, Player::Player2, params);

        score += (p2_pos_score - p1_pos_score) * params.position_weight / 10;
        score += p2_strategic_score - p1_strategic_score;
        score += self.evaluate_board_control_with_params(state, params);

        // Add new strategic evaluations
        score += self.evaluate_piece_coordination_with_params(state, params);
        score += self.evaluate_vulnerability_with_params(state, params);
        score += self.evaluate_blocking_with_params(state, params);
        score += self.evaluate_center_control_with_params(state, params);
        score += self.evaluate_rosette_chains_with_params(state, params);
        score += self.evaluate_early_game_with_params(state, params);
        score += self.evaluate_late_game_urgency_with_params(state, params);
        score += self.evaluate_turn_order_with_params(state, params);
        score += self.evaluate_mobility_with_params(state, params);
        score += self.evaluate_attack_pressure_with_params(state, params);
        score += self.evaluate_defensive_structure_with_params(state, params);

        score
    }

    fn evaluate_player_position_with_params(
        &self,
        state: &GameState,
        player: Player,
        params: &HeuristicParams,
    ) -> (i32, i32) {
        let pieces = state.get_pieces(player);
        let track = GameState::get_player_track(player);
        let (mut position_score, mut strategic_score) = (0i32, 0i32);

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                if let Some(track_pos) = track
                    .iter()
                    .position(|&square| square as i8 == piece.square)
                {
                    position_score += track_pos as i32 + 1;
                    if GameState::is_rosette(piece.square as u8) {
                        strategic_score += params.rosette_safety_bonus;
                    }
                    if track_pos >= 4 && track_pos <= 11 {
                        strategic_score += params.advancement_bonus;
                    }
                    if track_pos >= 12 {
                        strategic_score += params.advancement_bonus * 2;
                    }
                }
            }
        }
        (position_score, strategic_score)
    }

    fn evaluate_board_control_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let mut control_score = 0i32;
        for &rosette in &[0, 7, 13, 15, 16] {
            if let Some(occupant) = state.board[rosette as usize] {
                control_score += if occupant.player == Player::Player2 {
                    params.rosette_safety_bonus
                } else {
                    -params.rosette_safety_bonus
                };
            }
        }
        control_score
    }

    fn evaluate_piece_coordination_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_coordination = self.calculate_piece_coordination(state, Player::Player1);
        let p2_coordination = self.calculate_piece_coordination(state, Player::Player2);
        (p2_coordination - p1_coordination) * params.piece_coordination_bonus
    }

    fn evaluate_vulnerability_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_vulnerability = self.calculate_vulnerability(state, Player::Player1);
        let p2_vulnerability = self.calculate_vulnerability(state, Player::Player2);
        (p1_vulnerability - p2_vulnerability) * params.vulnerability_penalty
    }

    fn evaluate_blocking_with_params(&self, state: &GameState, params: &HeuristicParams) -> i32 {
        let p1_blocking = self.calculate_blocking(state, Player::Player1);
        let p2_blocking = self.calculate_blocking(state, Player::Player2);
        (p2_blocking - p1_blocking) * params.blocking_bonus
    }

    fn evaluate_rosette_chains_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_chains = self.calculate_rosette_chains(state, Player::Player1);
        let p2_chains = self.calculate_rosette_chains(state, Player::Player2);
        (p2_chains - p1_chains) * params.rosette_chain_bonus
    }

    fn evaluate_late_game_urgency_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
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

        if p1_finished >= 5 || p2_finished >= 5 {
            (p2_finished as i32 - p1_finished as i32) * params.late_game_urgency
        } else {
            0
        }
    }

    fn evaluate_turn_order_with_params(&self, state: &GameState, params: &HeuristicParams) -> i32 {
        if state.current_player == Player::Player2 {
            params.turn_order_bonus
        } else {
            -params.turn_order_bonus
        }
    }

    // Helper methods for calculations
    fn calculate_piece_coordination(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let mut coordination = 0;
        let mut on_board_pieces = Vec::new();

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                on_board_pieces.push(piece.square);
            }
        }

        for i in 0..on_board_pieces.len() {
            for j in (i + 1)..on_board_pieces.len() {
                let distance = (on_board_pieces[i] - on_board_pieces[j]).abs();
                if distance <= 3 {
                    coordination += 1;
                }
            }
        }
        coordination
    }

    fn calculate_vulnerability(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let mut vulnerability = 0;

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                if !GameState::is_rosette(piece.square as u8) {
                    // Check if piece can be captured by opponent
                    let opponent = player.opponent();
                    let opponent_pieces = state.get_pieces(opponent);
                    for opp_piece in opponent_pieces {
                        if opp_piece.square >= 0 && opp_piece.square < crate::BOARD_SIZE as i8 {
                            let distance = (piece.square - opp_piece.square).abs();
                            if distance <= 4 {
                                vulnerability += 1;
                            }
                        }
                    }
                }
            }
        }
        vulnerability
    }

    fn calculate_blocking(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let opponent = player.opponent();
        let opponent_track = GameState::get_player_track(opponent);
        let mut blocking = 0;

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                if let Some(track_pos) =
                    opponent_track.iter().position(|&s| s as i8 == piece.square)
                {
                    if track_pos > 0 {
                        blocking += 1;
                    }
                }
            }
        }
        blocking
    }

    fn calculate_center_control(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        pieces
            .iter()
            .filter(|p| p.square >= 4 && p.square <= 11)
            .count() as i32
    }

    fn calculate_rosette_chains(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let rosettes = [0, 7, 13, 15, 16];
        let mut rosette_count = 0;

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                if rosettes.contains(&(piece.square as u8)) {
                    rosette_count += 1;
                }
            }
        }
        rosette_count
    }

    fn evaluate_center_control_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_center = self.calculate_center_control(state, Player::Player1);
        let p2_center = self.calculate_center_control(state, Player::Player2);
        (p2_center - p1_center) * params.center_control_bonus
    }

    fn evaluate_early_game_with_params(&self, state: &GameState, params: &HeuristicParams) -> i32 {
        let p1_on_board = state
            .player1_pieces
            .iter()
            .filter(|p| p.square >= 0 && p.square < 20)
            .count();
        let p2_on_board = state
            .player2_pieces
            .iter()
            .filter(|p| p.square >= 0 && p.square < 20)
            .count();

        // Early game bonus for getting pieces on board
        if p1_on_board + p2_on_board < 8 {
            (p2_on_board as i32 - p1_on_board as i32) * params.early_game_bonus
        } else {
            0
        }
    }

    fn evaluate_mobility_with_params(&self, state: &GameState, params: &HeuristicParams) -> i32 {
        let p1_mobility = state.get_valid_moves().len();
        let p2_mobility = {
            let mut temp_state = state.clone();
            temp_state.current_player = Player::Player2;
            temp_state.get_valid_moves().len()
        };
        (p2_mobility as i32 - p1_mobility as i32) * params.mobility_bonus
    }

    fn evaluate_attack_pressure_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_pressure = self.calculate_attack_pressure(state, Player::Player1);
        let p2_pressure = self.calculate_attack_pressure(state, Player::Player2);
        (p2_pressure - p1_pressure) * params.attack_pressure_bonus
    }

    fn evaluate_defensive_structure_with_params(
        &self,
        state: &GameState,
        params: &HeuristicParams,
    ) -> i32 {
        let p1_defense = self.calculate_defensive_structure(state, Player::Player1);
        let p2_defense = self.calculate_defensive_structure(state, Player::Player2);
        (p2_defense - p1_defense) * params.defensive_structure_bonus
    }

    fn calculate_attack_pressure(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let opponent = player.opponent();
        let opponent_pieces = state.get_pieces(opponent);
        let mut pressure = 0;

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                for opponent_piece in opponent_pieces {
                    if opponent_piece.square >= 0 && opponent_piece.square < crate::BOARD_SIZE as i8
                    {
                        let distance = (piece.square - opponent_piece.square).abs();
                        if distance <= 4 {
                            pressure += (5 - distance) as i32;
                        }
                    }
                }
            }
        }
        pressure
    }

    fn calculate_defensive_structure(&self, state: &GameState, player: Player) -> i32 {
        let pieces = state.get_pieces(player);
        let mut defensive_score = 0;

        for piece in pieces {
            if piece.square >= 0 && piece.square < crate::BOARD_SIZE as i8 {
                if GameState::is_rosette(piece.square as u8) {
                    defensive_score += 3; // Rosettes are very safe
                } else {
                    defensive_score += 1; // Regular squares
                }
            }
        }
        defensive_score
    }
}

pub struct GeneticAlgorithm {
    population_size: usize,
    mutation_rate: f64,
    tournament_size: usize,
    games_per_individual: u32,
}

impl GeneticAlgorithm {
    pub fn new(
        population_size: usize,
        mutation_rate: f64,
        tournament_size: usize,
        games_per_individual: u32,
    ) -> Self {
        Self {
            population_size,
            mutation_rate,
            tournament_size,
            games_per_individual,
        }
    }

    pub fn evolve(&self, generations: usize) -> HeuristicParams {
        let mut population: Vec<GeneticIndividual> = Vec::with_capacity(self.population_size);

        // Start with a mix of default and random parameters for better exploration
        // Include the default parameters as a baseline
        population.push(GeneticIndividual::new(HeuristicParams::new()));

        // Add some mutated versions of default parameters
        let default_params = HeuristicParams::new();
        for _ in 0..(self.population_size / 4).max(1) {
            population.push(GeneticIndividual::new(default_params.mutate(0.3)));
        }

        // Fill the rest with random parameters
        while population.len() < self.population_size {
            population.push(GeneticIndividual::new(HeuristicParams::random()));
        }

        for generation in 0..generations {
            println!("Generation {}: Evaluating fitness...", generation);
            self.evaluate_population(&mut population);

            // Sort by fitness (best first)
            population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_individual = &population[0];
            println!(
                "Best fitness: {:.4}, win rate: {:.2}%",
                best_individual.fitness,
                best_individual.win_rate() * 100.0
            );

            if generation < generations - 1 {
                population = self.create_next_generation(&population);
            }
        }

        population[0].params.clone()
    }

    fn evaluate_population(&self, population: &mut Vec<GeneticIndividual>) {
        let population_size = population.len();

        let mut handles = vec![];

        for i in 0..population_size {
            let population_clone = population.clone();
            let games_per_individual = self.games_per_individual;

            let handle = thread::spawn(move || {
                let mut wins = 0;
                let mut games_played = 0;

                for j in 0..population_clone.len() {
                    if i != j {
                        for _ in 0..games_per_individual {
                            let result = Self::play_game(i, j, &population_clone);
                            games_played += 1;
                            if result == i {
                                wins += 1;
                            }
                        }
                    }
                }

                (i, wins, games_played)
            });

            handles.push(handle);
        }

        for handle in handles {
            if let Ok((i, wins, games_played)) = handle.join() {
                population[i].wins = wins;
                population[i].games_played = games_played;
                population[i].fitness = if games_played > 0 {
                    wins as f64 / games_played as f64
                } else {
                    0.0
                };
            }
        }
    }

    fn play_game(
        player1_idx: usize,
        player2_idx: usize,
        population: &Vec<GeneticIndividual>,
    ) -> usize {
        let player1_params = &population[player1_idx].params;
        let player2_params = &population[player2_idx].params;

        let mut state = GameState::new();
        let mut rng = thread_rng();

        while !state.is_game_over() {
            state.dice_roll = rng.gen_range(1..5);
            let valid_moves = state.get_valid_moves();

            if valid_moves.is_empty() {
                state.current_player = state.current_player.opponent();
                continue;
            }

            let mut ai = if state.current_player == Player::Player1 {
                GeneticAI::new(player1_params.clone())
            } else {
                GeneticAI::new(player2_params.clone())
            };

            let (best_move, _) = ai.get_best_move(&state);

            if let Some(move_idx) = best_move {
                state.make_move(move_idx).unwrap();
            } else {
                state.current_player = state.current_player.opponent();
            }
        }

        if state
            .player1_pieces
            .iter()
            .filter(|p| p.square == 20)
            .count()
            == PIECES_PER_PLAYER
        {
            player1_idx
        } else {
            player2_idx
        }
    }

    fn create_next_generation(&self, population: &[GeneticIndividual]) -> Vec<GeneticIndividual> {
        let mut new_population = Vec::with_capacity(self.population_size);

        // Elitism: keep the best individual
        new_population.push(GeneticIndividual::new(population[0].params.clone()));

        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection(population);
            let parent2 = self.tournament_selection(population);

            let mut child_params = parent1.params.crossover(&parent2.params);
            child_params = child_params.mutate(self.mutation_rate);

            new_population.push(GeneticIndividual::new(child_params));
        }

        new_population
    }

    fn tournament_selection<'a>(
        &self,
        population: &'a [GeneticIndividual],
    ) -> &'a GeneticIndividual {
        let mut rng = thread_rng();
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..self.tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.fitness > best.fitness {
                best = candidate;
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PiecePosition;

    fn create_test_params() -> HeuristicParams {
        HeuristicParams {
            win_score: 10000,
            finished_piece_value: 1000,
            position_weight: 15,
            advancement_bonus: 5,
            rosette_safety_bonus: 25,
            rosette_chain_bonus: 12,
            capture_bonus: 35,
            vulnerability_penalty: 20,
            center_control_bonus: 8,
            piece_coordination_bonus: 10,
            blocking_bonus: 15,
            early_game_bonus: 10,
            late_game_urgency: 30,
            turn_order_bonus: 5,
            mobility_bonus: 10,
            attack_pressure_bonus: 10,
            defensive_structure_bonus: 10,
        }
    }

    fn create_test_state() -> GameState {
        let mut state = GameState::new();
        state.current_player = Player::Player1;
        state.dice_roll = 3;
        state
    }

    #[test]
    fn test_piece_coordination_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no coordination
        assert_eq!(ai.calculate_piece_coordination(&state, Player::Player1), 0);
        assert_eq!(ai.calculate_piece_coordination(&state, Player::Player2), 0);

        // Place pieces close together (distance <= 3)
        state.player1_pieces[0].square = 4;
        state.player1_pieces[1].square = 5;
        state.player1_pieces[2].square = 6;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player1,
        });
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player1,
        });

        // Should have 3 coordinated pairs: (4,5), (4,6), (5,6)
        assert_eq!(ai.calculate_piece_coordination(&state, Player::Player1), 3);

        // Place pieces far apart (distance > 3)
        state.player1_pieces[3].square = 10;
        state.board[10] = Some(PiecePosition {
            square: 10,
            player: Player::Player1,
        });
        // Should still have 3 coordinated pairs, no additional coordination
        assert_eq!(ai.calculate_piece_coordination(&state, Player::Player1), 3);
    }

    #[test]
    fn test_vulnerability_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no vulnerability
        assert_eq!(ai.calculate_vulnerability(&state, Player::Player1), 0);

        // Place player1 piece on non-rosette square
        state.player1_pieces[0].square = 4;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });

        // Place player2 piece within capture range (distance <= 4)
        state.player2_pieces[0].square = 6;
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player2,
        });

        // Player1 piece should be vulnerable
        assert_eq!(ai.calculate_vulnerability(&state, Player::Player1), 1);

        // Place player1 piece on rosette - should not be vulnerable
        state.player1_pieces[1].square = 0; // Rosette square
        state.board[0] = Some(PiecePosition {
            square: 0,
            player: Player::Player1,
        });
        assert_eq!(ai.calculate_vulnerability(&state, Player::Player1), 1); // Still only 1 vulnerable piece
    }

    #[test]
    fn test_blocking_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no blocking
        assert_eq!(ai.calculate_blocking(&state, Player::Player1), 0);

        // Place player1 piece on player2's track (not at start)
        state.player1_pieces[0].square = 16; // Player2 track position
        state.board[16] = Some(PiecePosition {
            square: 16,
            player: Player::Player1,
        });

        // Should be blocking player2
        assert_eq!(ai.calculate_blocking(&state, Player::Player1), 1);

        // Place player1 piece on player2's start position - should not count as blocking
        state.player1_pieces[1].square = 19; // Player2 start
        state.board[19] = Some(PiecePosition {
            square: 19,
            player: Player::Player1,
        });
        assert_eq!(ai.calculate_blocking(&state, Player::Player1), 1); // Still only 1 blocking piece
    }

    #[test]
    fn test_center_control_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no center control
        assert_eq!(ai.calculate_center_control(&state, Player::Player1), 0);

        // Place pieces in center (squares 4-11)
        state.player1_pieces[0].square = 4;
        state.player1_pieces[1].square = 5;
        state.player1_pieces[2].square = 6;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player1,
        });
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player1,
        });

        assert_eq!(ai.calculate_center_control(&state, Player::Player1), 3);

        // Place piece outside center
        state.player1_pieces[3].square = 12;
        state.board[12] = Some(PiecePosition {
            square: 12,
            player: Player::Player1,
        });
        assert_eq!(ai.calculate_center_control(&state, Player::Player1), 3); // Still only 3 in center
    }

    #[test]
    fn test_rosette_chains_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no rosette chains
        assert_eq!(ai.calculate_rosette_chains(&state, Player::Player1), 0);

        // Place pieces on rosette squares [0, 7, 13, 15, 16]
        state.player1_pieces[0].square = 0;
        state.player1_pieces[1].square = 7;
        state.player1_pieces[2].square = 13;
        state.board[0] = Some(PiecePosition {
            square: 0,
            player: Player::Player1,
        });
        state.board[7] = Some(PiecePosition {
            square: 7,
            player: Player::Player1,
        });
        state.board[13] = Some(PiecePosition {
            square: 13,
            player: Player::Player1,
        });

        assert_eq!(ai.calculate_rosette_chains(&state, Player::Player1), 3);

        // Place piece on non-rosette square
        state.player1_pieces[3].square = 5;
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player1,
        });
        assert_eq!(ai.calculate_rosette_chains(&state, Player::Player1), 3); // Still only 3 on rosettes
    }

    #[test]
    fn test_attack_pressure_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no attack pressure
        assert_eq!(ai.calculate_attack_pressure(&state, Player::Player1), 0);

        // Place player1 piece near player2 piece
        state.player1_pieces[0].square = 4;
        state.player2_pieces[0].square = 6;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player2,
        });

        // Distance is 2, so pressure should be (5-2) = 3
        assert_eq!(ai.calculate_attack_pressure(&state, Player::Player1), 3);

        // Add another player2 piece at distance 4
        state.player2_pieces[1].square = 8;
        state.board[8] = Some(PiecePosition {
            square: 8,
            player: Player::Player2,
        });
        // Pressure should be 3 + (5-4) = 4
        assert_eq!(ai.calculate_attack_pressure(&state, Player::Player1), 4);
    }

    #[test]
    fn test_defensive_structure_calculation() {
        let ai = GeneticAI::new(create_test_params());
        let mut state = create_test_state();

        // Test empty board - no defensive structure
        assert_eq!(ai.calculate_defensive_structure(&state, Player::Player1), 0);

        // Place pieces on regular squares
        state.player1_pieces[0].square = 4;
        state.player1_pieces[1].square = 5;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player1,
        });

        // Should have 2 defensive points (1 each for regular squares)
        assert_eq!(ai.calculate_defensive_structure(&state, Player::Player1), 2);

        // Place piece on rosette square
        state.player1_pieces[2].square = 0; // Rosette
        state.board[0] = Some(PiecePosition {
            square: 0,
            player: Player::Player1,
        });
        // Should have 2 + 3 = 5 defensive points
        assert_eq!(ai.calculate_defensive_structure(&state, Player::Player1), 5);
    }

    #[test]
    fn test_evaluation_functions_with_params() {
        let ai = GeneticAI::new(create_test_params());
        let params = create_test_params();
        let mut state = create_test_state();

        // Test turn order evaluation (simplest test)
        state.current_player = Player::Player1;
        let turn_score_p1 = ai.evaluate_turn_order_with_params(&state, &params);
        state.current_player = Player::Player2;
        let turn_score_p2 = ai.evaluate_turn_order_with_params(&state, &params);
        assert_eq!(turn_score_p1, -params.turn_order_bonus);
        assert_eq!(turn_score_p2, params.turn_order_bonus);

        // Test piece coordination evaluation
        state.player1_pieces[0].square = 4;
        state.player1_pieces[1].square = 5;
        state.player2_pieces[0].square = 6;
        state.player2_pieces[1].square = 8;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player1,
        });
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player2,
        });
        state.board[8] = Some(PiecePosition {
            square: 8,
            player: Player::Player2,
        });

        // Both players have 1 coordinated pair, so difference should be 0
        let coordination_score = ai.evaluate_piece_coordination_with_params(&state, &params);
        assert_eq!(coordination_score, 0);

        // Test vulnerability evaluation (just test that function runs)
        let vulnerability_score = ai.evaluate_vulnerability_with_params(&state, &params);
        // Just test that the function runs without panicking
        assert!(vulnerability_score >= -1000 && vulnerability_score <= 1000); // Reasonable bounds

        // Test center control evaluation
        let center_score = ai.evaluate_center_control_with_params(&state, &params);
        // Both players have pieces in center (4,5,6,8), so difference should be 0
        assert_eq!(center_score, 0);

        // Test rosette chains evaluation
        let rosette_score = ai.evaluate_rosette_chains_with_params(&state, &params);
        // No pieces on rosettes, so score should be 0
        assert_eq!(rosette_score, 0);
    }

    #[test]
    fn test_early_game_evaluation() {
        let ai = GeneticAI::new(create_test_params());
        let params = create_test_params();
        let mut state = create_test_state();

        // Early game: few pieces on board
        state.player1_pieces[0].square = 4;
        state.player2_pieces[0].square = 5;
        state.player2_pieces[1].square = 6;
        state.board[4] = Some(PiecePosition {
            square: 4,
            player: Player::Player1,
        });
        state.board[5] = Some(PiecePosition {
            square: 5,
            player: Player::Player2,
        });
        state.board[6] = Some(PiecePosition {
            square: 6,
            player: Player::Player2,
        });

        // Player2 has 2 pieces on board, Player1 has 1
        // Total pieces on board = 3 < 8, so early game bonus applies
        let early_game_score = ai.evaluate_early_game_with_params(&state, &params);
        assert_eq!(early_game_score, (2 - 1) * params.early_game_bonus);

        // Late game: many pieces on board
        for i in 0..5 {
            state.player1_pieces[i].square = i as i8;
            state.player2_pieces[i].square = (i + 10) as i8;
            state.board[i] = Some(PiecePosition {
                square: i as i8,
                player: Player::Player1,
            });
            state.board[i + 10] = Some(PiecePosition {
                square: (i + 10) as i8,
                player: Player::Player2,
            });
        }

        // Total pieces on board = 10 >= 8, so no early game bonus
        let late_early_game_score = ai.evaluate_early_game_with_params(&state, &params);
        assert_eq!(late_early_game_score, 0);
    }

    #[test]
    fn test_late_game_urgency_evaluation() {
        let ai = GeneticAI::new(create_test_params());
        let params = create_test_params();
        let mut state = create_test_state();

        // Early game: few finished pieces
        state.player1_pieces[0].square = 20; // Finished
        state.player2_pieces[0].square = 20; // Finished
        state.player2_pieces[1].square = 20; // Finished

        // Player1 has 1 finished, Player2 has 2 finished
        // But total finished < 5, so no late game urgency
        let early_urgency_score = ai.evaluate_late_game_urgency_with_params(&state, &params);
        assert_eq!(early_urgency_score, 0);

        // Late game: many finished pieces
        for i in 0..5 {
            state.player1_pieces[i].square = 20; // Finished
        }
        for i in 0..6 {
            state.player2_pieces[i].square = 20; // Finished
        }

        // Player1 has 5 finished, Player2 has 6 finished
        // Total finished >= 5, so late game urgency applies
        let late_urgency_score = ai.evaluate_late_game_urgency_with_params(&state, &params);
        assert_eq!(late_urgency_score, (6 - 5) * params.late_game_urgency);
    }

    #[test]
    fn test_mobility_evaluation() {
        let ai = GeneticAI::new(create_test_params());
        let params = create_test_params();
        let mut state = create_test_state();

        // Test that mobility evaluation function exists and runs without error
        let mobility_score = ai.evaluate_mobility_with_params(&state, &params);
        // Just test that the function runs without panicking
        assert!(mobility_score >= -1000 && mobility_score <= 1000); // Reasonable bounds
    }

    #[test]
    fn test_board_control_evaluation() {
        let ai = GeneticAI::new(create_test_params());
        let params = create_test_params();
        let mut state = create_test_state();

        // Test rosette control
        state.board[0] = Some(PiecePosition {
            square: 0,
            player: Player::Player1,
        }); // Rosette
        state.board[7] = Some(PiecePosition {
            square: 7,
            player: Player::Player2,
        }); // Rosette

        let control_score = ai.evaluate_board_control_with_params(&state, &params);
        // Player1 controls 1 rosette, Player2 controls 1 rosette
        // Score should be -rosette_safety_bonus + rosette_safety_bonus = 0
        assert_eq!(control_score, 0);

        // Player2 controls both rosettes
        state.board[0] = Some(PiecePosition {
            square: 0,
            player: Player::Player2,
        });
        let control_score_2 = ai.evaluate_board_control_with_params(&state, &params);
        // Player2 controls 2 rosettes, Player1 controls 0
        // Score should be 2 * rosette_safety_bonus
        assert_eq!(control_score_2, 2 * params.rosette_safety_bonus);
    }

    #[test]
    fn test_parameter_bounds() {
        let params = HeuristicParams::new();

        // Test that all parameters are within reasonable bounds
        assert!(params.win_score >= 1000 && params.win_score <= 20000);
        assert!(params.finished_piece_value >= 100 && params.finished_piece_value <= 3000);
        assert!(params.position_weight >= 1 && params.position_weight <= 100);
        assert!(params.advancement_bonus >= 1 && params.advancement_bonus <= 50);
        assert!(params.rosette_safety_bonus >= 5 && params.rosette_safety_bonus <= 100);
        assert!(params.rosette_chain_bonus >= 1 && params.rosette_chain_bonus <= 50);
        assert!(params.capture_bonus >= 10 && params.capture_bonus <= 100);
        assert!(params.vulnerability_penalty >= 5 && params.vulnerability_penalty <= 80);
        assert!(params.center_control_bonus >= 1 && params.center_control_bonus <= 40);
        assert!(params.piece_coordination_bonus >= 1 && params.piece_coordination_bonus <= 50);
        assert!(params.blocking_bonus >= 1 && params.blocking_bonus <= 60);
        assert!(params.early_game_bonus >= 1 && params.early_game_bonus <= 30);
        assert!(params.late_game_urgency >= 5 && params.late_game_urgency <= 100);
        assert!(params.turn_order_bonus >= 1 && params.turn_order_bonus <= 30);
        assert!(params.mobility_bonus >= 1 && params.mobility_bonus <= 30);
        assert!(params.attack_pressure_bonus >= 1 && params.attack_pressure_bonus <= 30);
        assert!(params.defensive_structure_bonus >= 1 && params.defensive_structure_bonus <= 30);
    }

    #[test]
    fn test_mutation_preserves_bounds() {
        let params = HeuristicParams::new();
        let mutated = params.mutate(1.0); // 100% mutation rate

        // Test that mutated parameters stay within bounds
        assert!(mutated.win_score >= 1000 && mutated.win_score <= 20000);
        assert!(mutated.finished_piece_value >= 100 && mutated.finished_piece_value <= 3000);
        assert!(mutated.position_weight >= 1 && mutated.position_weight <= 100);
        assert!(mutated.advancement_bonus >= 1 && mutated.advancement_bonus <= 50);
        assert!(mutated.rosette_safety_bonus >= 5 && mutated.rosette_safety_bonus <= 100);
        assert!(mutated.rosette_chain_bonus >= 1 && mutated.rosette_chain_bonus <= 50);
        assert!(mutated.capture_bonus >= 10 && mutated.capture_bonus <= 100);
        assert!(mutated.vulnerability_penalty >= 5 && mutated.vulnerability_penalty <= 80);
        assert!(mutated.center_control_bonus >= 1 && mutated.center_control_bonus <= 40);
        assert!(mutated.piece_coordination_bonus >= 1 && mutated.piece_coordination_bonus <= 50);
        assert!(mutated.blocking_bonus >= 1 && mutated.blocking_bonus <= 60);
        assert!(mutated.early_game_bonus >= 1 && mutated.early_game_bonus <= 30);
        assert!(mutated.late_game_urgency >= 5 && mutated.late_game_urgency <= 100);
        assert!(mutated.turn_order_bonus >= 1 && mutated.turn_order_bonus <= 30);
        assert!(mutated.mobility_bonus >= 1 && mutated.mobility_bonus <= 30);
        assert!(mutated.attack_pressure_bonus >= 1 && mutated.attack_pressure_bonus <= 30);
        assert!(mutated.defensive_structure_bonus >= 1 && mutated.defensive_structure_bonus <= 30);
    }
}
