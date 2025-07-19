use crate::{GameState, Player, PIECES_PER_PLAYER};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::thread;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeuristicParams {
    pub win_score: i32,
    pub finished_piece_value: i32,
    pub position_weight: i32,
    pub safety_bonus: i32,
    pub rosette_control_bonus: i32,
    pub advancement_bonus: i32,
    pub capture_bonus: i32,
    pub center_lane_bonus: i32,
}

impl HeuristicParams {
    pub fn new() -> Self {
        Self {
            win_score: 10000,
            finished_piece_value: 1000,
            position_weight: 15,
            safety_bonus: 25,
            rosette_control_bonus: 40,
            advancement_bonus: 5,
            capture_bonus: 35,
            center_lane_bonus: 2,
        }
    }

    pub fn random() -> Self {
        let mut rng = thread_rng();
        Self {
            win_score: rng.gen_range(5000..15000),
            finished_piece_value: rng.gen_range(500..2000),
            position_weight: rng.gen_range(5..30),
            safety_bonus: rng.gen_range(10..50),
            rosette_control_bonus: rng.gen_range(20..80),
            advancement_bonus: rng.gen_range(1..15),
            capture_bonus: rng.gen_range(20..60),
            center_lane_bonus: rng.gen_range(1..10),
        }
    }

    pub fn mutate(&self, mutation_rate: f64) -> Self {
        let mut rng = thread_rng();
        let mut new_params = self.clone();

        if rng.gen_bool(mutation_rate) {
            new_params.win_score = (new_params.win_score as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.finished_piece_value =
                (new_params.finished_piece_value as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.position_weight =
                (new_params.position_weight as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.safety_bonus =
                (new_params.safety_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.rosette_control_bonus =
                (new_params.rosette_control_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.advancement_bonus =
                (new_params.advancement_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.capture_bonus =
                (new_params.capture_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }
        if rng.gen_bool(mutation_rate) {
            new_params.center_lane_bonus =
                (new_params.center_lane_bonus as f64 * rng.gen_range(0.8..1.2)) as i32;
        }

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
            safety_bonus: if rng.gen_bool(0.5) {
                self.safety_bonus
            } else {
                other.safety_bonus
            },
            rosette_control_bonus: if rng.gen_bool(0.5) {
                self.rosette_control_bonus
            } else {
                other.rosette_control_bonus
            },
            advancement_bonus: if rng.gen_bool(0.5) {
                self.advancement_bonus
            } else {
                other.advancement_bonus
            },
            capture_bonus: if rng.gen_bool(0.5) {
                self.capture_bonus
            } else {
                other.capture_bonus
            },
            center_lane_bonus: if rng.gen_bool(0.5) {
                self.center_lane_bonus
            } else {
                other.center_lane_bonus
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
                        strategic_score += params.safety_bonus;
                    }
                    if track_pos >= 4 && track_pos <= 11 {
                        strategic_score += params.advancement_bonus + params.center_lane_bonus;
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
                    params.rosette_control_bonus
                } else {
                    -params.rosette_control_bonus
                };
            }
        }
        control_score
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
        let mut population: Vec<GeneticIndividual> = (0..self.population_size)
            .map(|_| GeneticIndividual::new(HeuristicParams::random()))
            .collect();

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

        fn tournament_selection<'a>(&self, population: &'a [GeneticIndividual]) -> &'a GeneticIndividual {
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
