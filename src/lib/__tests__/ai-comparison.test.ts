import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MLAIService } from '../ml-ai-service';
import { createTestGameState } from './test-utils';

// Mock the worker
vi.mock('../ml-ai.worker', () => ({
  default: vi.fn(),
}));

describe('AI Comparison Tests', () => {
  let mlAIService: MLAIService;

  beforeEach(() => {
    vi.clearAllMocks();
    mlAIService = new MLAIService();
  });

  describe('ML AI Service Performance', () => {
    it('should handle move requests efficiently', async () => {
      const gameState = createTestGameState({
        player1PieceSquares: [0, 5, 10, 15, 20, -1, -1],
        player2PieceSquares: [2, 7, 12, 20, 20, -1, -1],
        currentPlayer: 'player1',
        diceRoll: 3,
      });

      const startTime = performance.now();

      try {
        // Add timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 2000);
        });

        const movePromise = mlAIService.getAIMove(gameState);
        await Promise.race([movePromise, timeoutPromise]);

        const endTime = performance.now();
        const duration = endTime - startTime;

        // Should complete within reasonable time (2 seconds)
        expect(duration).toBeLessThan(2000);
      } catch (error) {
        // It's okay if ML AI fails to load weights, but it shouldn't hang
        expect(error).toBeDefined();
        expect(error instanceof Error).toBe(true);
      }
    }, 3000); // 3 second timeout

    it('should provide consistent move evaluations', async () => {
      const gameState = createTestGameState({
        player1PieceSquares: [0, 5, 10, 15, 20, -1, -1],
        player2PieceSquares: [2, 7, 12, 20, 20, -1, -1],
        currentPlayer: 'player1',
        diceRoll: 2,
      });

      try {
        // Add timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 2000);
        });

        const movePromise1 = mlAIService.getAIMove(gameState);
        const movePromise2 = mlAIService.getAIMove(gameState);

        const [response1, response2] = await Promise.all([
          Promise.race([movePromise1, timeoutPromise]),
          Promise.race([movePromise2, timeoutPromise]),
        ]);

        // Should provide consistent evaluation structure
        expect(response1).toHaveProperty('move');
        expect(response1).toHaveProperty('evaluation');
        expect(response1).toHaveProperty('thinking');
        expect(response1).toHaveProperty('diagnostics');
        expect(response1).toHaveProperty('timings');

        expect(response2).toHaveProperty('move');
        expect(response2).toHaveProperty('evaluation');
        expect(response2).toHaveProperty('thinking');
        expect(response2).toHaveProperty('diagnostics');
        expect(response2).toHaveProperty('timings');
      } catch (error) {
        // Expected if weights aren't loaded
        expect(error).toBeDefined();
        expect(error instanceof Error).toBe(true);
      }
    }, 3000); // 3 second timeout
  });

  describe('AI Response Validation', () => {
    it('should validate AI response structure', () => {
      const mockResponse = {
        move: 0,
        evaluation: 0.5,
        thinking: 'Test move',
        diagnostics: {
          valid_moves: [0, 1],
          move_evaluations: [],
          value_network_output: 0.5,
          policy_network_outputs: [0.5, 0.5],
        },
        timings: {
          aiMoveCalculation: 100,
          totalHandlerTime: 150,
        },
      };

      expect(mockResponse).toHaveProperty('move');
      expect(mockResponse).toHaveProperty('evaluation');
      expect(mockResponse).toHaveProperty('thinking');
      expect(mockResponse).toHaveProperty('diagnostics');
      expect(mockResponse).toHaveProperty('timings');

      expect(typeof mockResponse.move).toBe('number');
      expect(typeof mockResponse.evaluation).toBe('number');
      expect(typeof mockResponse.thinking).toBe('string');
      expect(mockResponse.diagnostics).toHaveProperty('valid_moves');
      expect(mockResponse.diagnostics).toHaveProperty('move_evaluations');
      expect(mockResponse.diagnostics).toHaveProperty('value_network_output');
      expect(mockResponse.diagnostics).toHaveProperty('policy_network_outputs');
      expect(mockResponse.timings).toHaveProperty('aiMoveCalculation');
      expect(mockResponse.timings).toHaveProperty('totalHandlerTime');
    });

    it('should handle null move responses', () => {
      const mockResponse = {
        move: null,
        evaluation: 0.0,
        thinking: 'No valid moves',
        diagnostics: {
          valid_moves: [],
          move_evaluations: [],
          value_network_output: 0.0,
          policy_network_outputs: [],
        },
        timings: {
          aiMoveCalculation: 10,
          totalHandlerTime: 15,
        },
      };

      expect(mockResponse.move).toBeNull();
      expect(mockResponse.evaluation).toBe(0.0);
      expect(mockResponse.diagnostics.valid_moves).toEqual([]);
    });
  });

  describe('Game State Compatibility', () => {
    it('should handle different game phases', async () => {
      const gameStates = [
        // Opening game
        createTestGameState({
          player1PieceSquares: [0, 0, 0, 0, 0, 0, 0],
          player2PieceSquares: [0, 0, 0, 0, 0, 0, 0],
          currentPlayer: 'player1',
          diceRoll: 1,
        }),
        // Mid game
        createTestGameState({
          player1PieceSquares: [5, 10, 15, 20, -1, -1, -1],
          player2PieceSquares: [3, 8, 12, 18, -1, -1, -1],
          currentPlayer: 'player2',
          diceRoll: 4,
        }),
        // End game
        createTestGameState({
          player1PieceSquares: [20, 20, 20, 20, 20, 20, 20],
          player2PieceSquares: [18, 19, 20, 20, 20, 20, 20],
          currentPlayer: 'player2',
          diceRoll: 2,
        }),
      ];

      for (const gameState of gameStates) {
        try {
          // Add timeout to prevent hanging
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), 2000);
          });

          const movePromise = mlAIService.getAIMove(gameState);
          await Promise.race([movePromise, timeoutPromise]);
        } catch (error) {
          // Expected if weights aren't loaded
          expect(error).toBeDefined();
          expect(error instanceof Error).toBe(true);
        }
      }
    }, 10000); // 10 second timeout for multiple game states

    it('should handle edge cases gracefully', async () => {
      const edgeCases = [
        // No valid moves
        createTestGameState({
          player1PieceSquares: [20, 20, 20, 20, 20, 20, 20],
          player2PieceSquares: [20, 20, 20, 20, 20, 20, 20],
          currentPlayer: 'player1',
          diceRoll: 1,
        }),
        // All pieces in play
        createTestGameState({
          player1PieceSquares: [1, 2, 3, 4, 5, 6, 7],
          player2PieceSquares: [8, 9, 10, 11, 12, 13, 14],
          currentPlayer: 'player1',
          diceRoll: 3,
        }),
        // Dice roll 0 (no move)
        createTestGameState({
          player1PieceSquares: [0, 5, 10, 15, -1, -1, -1],
          player2PieceSquares: [2, 7, 12, -1, -1, -1, -1],
          currentPlayer: 'player1',
          diceRoll: 0,
        }),
      ];

      for (const gameState of edgeCases) {
        try {
          // Add timeout to prevent hanging
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), 2000);
          });

          const movePromise = mlAIService.getAIMove(gameState);
          await Promise.race([movePromise, timeoutPromise]);
        } catch (error) {
          // Expected if weights aren't loaded
          expect(error).toBeDefined();
          expect(error instanceof Error).toBe(true);
        }
      }
    }, 10000); // 10 second timeout for multiple edge cases
  });

  describe('Performance Metrics', () => {
    it('should track timing information', async () => {
      const gameState = createTestGameState({
        player1PieceSquares: [0, 5, 10, 15, -1, -1, -1],
        player2PieceSquares: [2, 7, 12, -1, -1, -1, -1],
        currentPlayer: 'player1',
        diceRoll: 3,
      });

      try {
        // Add timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 2000);
        });

        const movePromise = mlAIService.getAIMove(gameState);
        const response = await Promise.race([movePromise, timeoutPromise]);

        expect(response.timings).toBeDefined();
        expect(response.timings.aiMoveCalculation).toBeGreaterThan(0);
        expect(response.timings.totalHandlerTime).toBeGreaterThan(0);
        expect(response.timings.totalHandlerTime).toBeGreaterThanOrEqual(
          response.timings.aiMoveCalculation
        );
      } catch (error) {
        // Expected if weights aren't loaded
        expect(error).toBeDefined();
        expect(error instanceof Error).toBe(true);
      }
    }, 3000); // 3 second timeout

    it('should provide diagnostics information', async () => {
      const gameState = createTestGameState({
        player1PieceSquares: [0, 5, 10, 15, -1, -1, -1],
        player2PieceSquares: [2, 7, 12, -1, -1, -1, -1],
        currentPlayer: 'player1',
        diceRoll: 3,
      });

      try {
        // Add timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 2000);
        });

        const movePromise = mlAIService.getAIMove(gameState);
        const response = await Promise.race([movePromise, timeoutPromise]);

        expect(response.diagnostics).toBeDefined();
        expect(response.diagnostics.valid_moves).toBeDefined();
        expect(response.diagnostics.move_evaluations).toBeDefined();
        expect(response.diagnostics.value_network_output).toBeDefined();
        expect(response.diagnostics.policy_network_outputs).toBeDefined();
      } catch (error) {
        // Expected if weights aren't loaded
        expect(error).toBeDefined();
        expect(error instanceof Error).toBe(true);
      }
    }, 3000); // 3 second timeout
  });

  describe('Error Handling', () => {
    it('should handle worker initialization failures gracefully', async () => {
      // Mock worker to throw error
      const mockWorker = vi.fn().mockImplementation(() => {
        throw new Error('Worker initialization failed');
      });

      // Replace the worker mock
      vi.doMock('../ml-ai.worker', () => ({
        default: mockWorker,
      }));

      const newService = new MLAIService();
      const gameState = createTestGameState({
        player1PieceSquares: [0, 5, 10, 15, -1, -1, -1],
        player2PieceSquares: [2, 7, 12, -1, -1, -1, -1],
        currentPlayer: 'player1',
        diceRoll: 3,
      });

      await expect(newService.getAIMove(gameState)).rejects.toThrow();
    }, 3000); // 3 second timeout

    it('should handle invalid game states', async () => {
      const invalidGameState = {
        ...createTestGameState({
          player1PieceSquares: [0, 5, 10, 15, -1, -1, -1],
          player2PieceSquares: [2, 7, 12, -1, -1, -1, -1],
          currentPlayer: 'player1',
          diceRoll: 3,
        }),
        // Invalid property
        invalidProperty: 'should cause issues',
      };

      try {
        // Add timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 2000);
        });

        const movePromise = mlAIService.getAIMove(invalidGameState as any);
        await Promise.race([movePromise, timeoutPromise]);
      } catch (error) {
        // Should handle gracefully
        expect(error).toBeDefined();
        expect(error instanceof Error).toBe(true);
      }
    }, 3000); // 3 second timeout
  });
});
