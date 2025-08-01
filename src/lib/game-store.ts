import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { initializeGame, processDiceRoll, makeMove as makeMoveLogic } from './game-logic';
import { WasmAiService } from './wasm-ai-service';
import { MLAIService } from './ml-ai-service';
import { useStatsStore } from './stats-store';
import type { GameState, Player, MoveType, AIResponse } from './types';
import { saveGame } from './actions';
import { getPlayerId } from './utils';
import { useUIStore } from './ui-store';

const LATEST_VERSION = 1;

const wasmAiService = new WasmAiService();
const mlAiService = new MLAIService();

type GameStore = {
  gameState: GameState;
  aiThinking: boolean;
  lastAIDiagnostics: AIResponse | null;
  lastAIMoveDuration: number | null;
  lastMoveType: MoveType | null;
  lastMovePlayer: Player | null;
  actions: {
    initialize: (fromStorage?: boolean) => void;
    processDiceRoll: (roll?: number) => void;
    endTurn: () => void;
    makeMove: (pieceIndex: number) => void;
    makeAIMove: (aiSource: 'heuristic' | 'client' | 'ml', isPlayer1AI?: boolean) => Promise<void>;
    reset: () => void;
    postGameToServer: () => Promise<void>;
    createNearWinningState: () => void;
  };
};

export const useGameStore = create<GameStore>()(
  persist(
    immer((set, get) => ({
      gameState: { ...initializeGame(), startTime: Date.now() },
      aiThinking: false,
      lastAIDiagnostics: null,
      lastAIMoveDuration: null,
      lastMoveType: null,
      lastMovePlayer: null,
      actions: {
        initialize: (fromStorage = false) => {
          if (!fromStorage) {
            set(state => {
              state.gameState = { ...initializeGame(), startTime: Date.now() };
              state.aiThinking = false;
              state.lastAIDiagnostics = null;
              state.lastAIMoveDuration = null;
              state.lastMoveType = null;
              state.lastMovePlayer = null;
            });
          }
        },
        processDiceRoll: roll => {
          const { gameState } = get();
          const newState = processDiceRoll(gameState, roll);
          set(state => {
            state.gameState = newState;
          });
        },
        endTurn: () => {
          set(state => {
            state.gameState.currentPlayer =
              state.gameState.currentPlayer === 'player1' ? 'player2' : 'player1';
            state.gameState.diceRoll = null;
            state.gameState.canMove = false;
            state.gameState.validMoves = [];
          });
        },
        makeMove: (pieceIndex: number) => {
          const { gameState } = get();
          if (gameState.canMove && gameState.validMoves.includes(pieceIndex)) {
            const [newState, moveType, movePlayer] = makeMoveLogic(gameState, pieceIndex);
            set(state => {
              state.gameState = newState;
              state.lastMoveType = moveType;
              state.lastMovePlayer = movePlayer;

              if (newState.gameStatus === 'finished') {
                if (newState.winner === 'player1') {
                  useStatsStore.getState().actions.incrementWins();
                } else {
                  useStatsStore.getState().actions.incrementLosses();
                }
              }
            });
          }
        },
        makeAIMove: async (aiSource: 'heuristic' | 'client' | 'ml', isPlayer1AI = false) => {
          const { gameState, actions } = get();

          if (!gameState.canMove) return;

          if (!isPlayer1AI && gameState.currentPlayer !== 'player2') {
            return;
          }

          console.log('GameStore: Starting AI move with source:', aiSource);
          console.log('GameStore: Current game state:', {
            currentPlayer: gameState.currentPlayer,
            diceRoll: gameState.diceRoll,
            validMoves: gameState.validMoves,
            canMove: gameState.canMove,
          });

          if (gameState.validMoves.length === 0) {
            console.log('GameStore: No valid moves, switching turn');
            set(state => {
              state.gameState = processDiceRoll({
                ...state.gameState,
                currentPlayer: 'player1',
                diceRoll: null,
                canMove: false,
                validMoves: [],
              });
              state.aiThinking = false;
            });
            return;
          }

          set(state => {
            state.aiThinking = true;
          });

          const startTime = performance.now();

          try {
            let aiResponse;

            if (aiSource === 'ml') {
              console.log('GameStore: Using ML AI service');
              const mlResponse = await mlAiService.getAIMove(gameState);
              console.log('GameStore: ML AI response received:', mlResponse);
              aiResponse = {
                move: mlResponse.move,
                evaluation: Math.round(mlResponse.evaluation * 1000),
                thinking: mlResponse.thinking,
                timings: {
                  aiMoveCalculation: mlResponse.timings?.aiMoveCalculation || 0,
                  totalHandlerTime: mlResponse.timings?.totalHandlerTime || 0,
                },
                diagnostics: {
                  searchDepth: 4,
                  validMoves: mlResponse.diagnostics.valid_moves,
                  moveEvaluations: mlResponse.diagnostics.move_evaluations.map(e => ({
                    pieceIndex: e.piece_index,
                    score: e.score,
                    moveType: e.move_type,
                    fromSquare: e.from_square,
                    toSquare: e.to_square ?? null,
                  })),
                  transpositionHits: 0,
                  nodesEvaluated: 1,
                },
                aiType: 'ml' as const,
              };
              console.log('GameStore: Processed ML AI response:', aiResponse);
            } else if (aiSource === 'heuristic') {
              console.log('GameStore: Using Heuristic AI service');
              const heuristicResponse = await wasmAiService.getHeuristicAIMove(gameState);
              console.log('GameStore: Heuristic AI response received:', heuristicResponse);
              aiResponse = { ...heuristicResponse, aiType: 'heuristic' as const };
            } else {
              console.log('GameStore: Using WASM AI service for', aiSource);
              const wasmResponse = await wasmAiService.getAIMove(gameState);
              console.log('GameStore: WASM AI response received:', wasmResponse);
              aiResponse = { ...wasmResponse, aiType: 'client' as const };
            }

            const duration = performance.now() - startTime;

            console.log('GameStore: Setting AI diagnostics:', aiResponse);
            set(state => {
              state.lastAIMoveDuration = duration;
              state.lastAIDiagnostics = aiResponse;
            });

            const { move: aiMove } = aiResponse;

            if (aiMove === null || aiMove === undefined || !gameState.validMoves.includes(aiMove)) {
              console.log('GameStore: Invalid AI move, using fallback');
              if (gameState.validMoves.length > 0) {
                actions.makeMove(gameState.validMoves[0]);
              }
            } else {
              console.log('GameStore: Making AI move:', aiMove);
              actions.makeMove(aiMove);
            }
          } catch (error) {
            console.error('GameStore: AI move failed:', error);
            if (gameState.validMoves.length > 0) {
              const fallbackMove =
                gameState.validMoves[Math.floor(Math.random() * gameState.validMoves.length)];
              console.warn('GameStore: Using fallback random move:', fallbackMove);
              actions.makeMove(fallbackMove);
            }
          } finally {
            set(state => {
              state.aiThinking = false;
            });
          }
        },
        reset: () => {
          set(state => {
            state.gameState = { ...initializeGame(), startTime: Date.now() };
            state.aiThinking = false;
            state.lastAIDiagnostics = null;
            state.lastAIMoveDuration = null;
            state.lastMoveType = null;
            state.lastMovePlayer = null;
          });
        },
        createNearWinningState: () => {
          set(state => {
            for (let i = 0; i < 6; i++) {
              state.gameState.player1Pieces[i].square = 20;
            }
            state.gameState.player1Pieces[6].square = 12;
            state.gameState.board[12] = state.gameState.player1Pieces[6];

            state.gameState.currentPlayer = 'player1';
            state.gameState.gameStatus = 'playing';
            state.gameState.winner = null;
            state.gameState.diceRoll = null;
            state.gameState.canMove = false;
            state.gameState.validMoves = [];

            state.aiThinking = false;
            state.lastAIDiagnostics = null;
            state.lastAIMoveDuration = null;
            state.lastMoveType = null;
            state.lastMovePlayer = null;
          });
        },
        postGameToServer: async () => {
          const { gameState } = get();
          if (gameState.gameStatus !== 'finished' || !gameState.winner) {
            return;
          }

          try {
            const duration = gameState.startTime ? Date.now() - gameState.startTime : undefined;
            let clientHeader = 'unknown';
            if (typeof window !== 'undefined' && window.navigator) {
              clientHeader = window.navigator.userAgent;
            }

            const uiStore = useUIStore.getState();
            const gameMode = uiStore.selectedMode || 'classic';

            const payload = {
              winner: gameState.winner,
              history: gameState.history,
              playerId: getPlayerId(),
              moveCount: gameState.history.length,
              duration,
              clientHeader,
              gameType: gameMode,
            };

            await saveGame(payload);
          } catch (error) {
            console.error('Failed to save game result:', error);
          }
        },
      },
    })),
    {
      name: 'rgou-game-storage',
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state, error) => {
        if (error) {
          console.error('Failed to rehydrate game store:', error);
        }
        if (state) {
          state.actions.initialize(true);
        }
      },
      version: LATEST_VERSION,
      migrate: (persistedState, version) => {
        const state = persistedState as Partial<GameStore>;
        if (version < LATEST_VERSION || !state || !state.gameState) {
          return { gameState: initializeGame() };
        }
        return { gameState: state.gameState };
      },
      partialize: state => ({
        gameState: state.gameState,
      }),
    }
  )
);

export const useGameStoreActions = () => useGameStore(state => state.actions);

export const useGameState = () => useGameStore(state => state.gameState);
export const useGameActions = () => useGameStore(state => state.actions);
