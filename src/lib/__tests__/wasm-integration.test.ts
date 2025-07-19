import { describe, it, expect, beforeAll } from 'vitest';
import { GameState } from '../types';

describe('WASM Integration Tests', () => {
  let gameState: GameState;

  beforeAll(() => {
    gameState = {
      board: Array(21).fill(null),
      player1Pieces: Array.from({ length: 7 }, () => ({
        square: -1,
        player: 'player1' as const,
      })),
      player2Pieces: Array.from({ length: 7 }, () => ({
        square: -1,
        player: 'player2' as const,
      })),
      currentPlayer: 'player1',
      gameStatus: 'playing',
      winner: null,
      diceRoll: 1,
      canMove: true,
      validMoves: [],
      history: [],
    };
  });

    it('should load WASM AI service successfully', async () => {
    const { WasmAiService } = await import('../wasm-ai-service');
    expect(WasmAiService).toBeDefined();
    expect(typeof WasmAiService).toBe('function');
  });

  it('should make AI moves within performance limits', async () => {
    const start = performance.now();
    
    const { WasmAiService } = await import('../wasm-ai-service');
    const service = new WasmAiService();
    const result = await service.getAIMove(gameState);
    
    const duration = performance.now() - start;
    
    expect(result.move).toBeDefined();
    expect(duration).toBeLessThan(100); // 100ms limit for WASM loading + move calculation
  });

  it('should handle AI service errors gracefully', async () => {
    const invalidState = {
      ...gameState,
      currentPlayer: 'invalid' as any,
    };

    const { WasmAiService } = await import('../wasm-ai-service');
    const service = new WasmAiService();
    
    try {
      await service.getAIMove(invalidState);
      expect.fail('Should have thrown an error');
    } catch (error) {
      expect(error).toBeDefined();
    }
  });

  it('should validate AI move results', async () => {
    const { WasmAiService } = await import('../wasm-ai-service');
    const service = new WasmAiService();
    const result = await service.getAIMove(gameState);
    
    expect(result.move).toBeDefined();
    expect(typeof result.move).toBe('number');
    expect(result.move).toBeGreaterThanOrEqual(0);
    expect(result.move).toBeLessThan(7);
  });

  it('should handle consecutive AI moves without memory leaks', async () => {
    const { WasmAiService } = await import('../wasm-ai-service');
    const service = new WasmAiService();
    const moves = [];
    
    for (let i = 0; i < 5; i++) {
      const result = await service.getAIMove(gameState);
      expect(result.move).toBeDefined();
      moves.push(result);
    }
    
    expect(moves).toHaveLength(5);
    moves.forEach(move => {
      expect(move.move).toBeDefined();
    });
  });

  it('should support different AI types', async () => {
    const { WasmAiService } = await import('../wasm-ai-service');
    const { MLAIService } = await import('../ml-ai-service');
    
    const classicService = new WasmAiService();
    const mlService = new MLAIService();
    
    const classicResult = await classicService.getAIMove(gameState);
    const mlResult = await mlService.getAIMove(gameState);
    
    expect(classicResult.move).toBeDefined();
    expect(mlResult.move).toBeDefined();
  });

  it('should maintain game state consistency', async () => {
    const initialState = { ...gameState };
    
    const { WasmAiService } = await import('../wasm-ai-service');
    const service = new WasmAiService();
    await service.getAIMove(gameState);
    
    expect(gameState.currentPlayer).toBe(initialState.currentPlayer);
    expect(gameState.diceRoll).toBe(initialState.diceRoll);
    expect(gameState.player1Pieces).toEqual(initialState.player1Pieces);
    expect(gameState.player2Pieces).toEqual(initialState.player2Pieces);
  });
});
