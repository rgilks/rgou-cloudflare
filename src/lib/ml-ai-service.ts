import type { GameState } from './schemas';
import pako from 'pako';

interface MLWeights {
  value_weights: number[];
  policy_weights: number[];
  value_network_config?: {
    input_size: number;
    hidden_sizes: number[];
    output_size: number;
  };
  policy_network_config?: {
    input_size: number;
    hidden_sizes: number[];
    output_size: number;
  };
}

interface MLResponse {
  move: number | null;
  evaluation: number;
  thinking: string;
  diagnostics: {
    valid_moves: number[];
    move_evaluations: Array<{
      piece_index: number;
      score: number;
      move_type: string;
      from_square: number;
      to_square?: number;
    }>;
    value_network_output: number;
    policy_network_outputs: number[];
  };
  timings: {
    aiMoveCalculation: number;
    totalHandlerTime: number;
  };
}

type PendingRequestType =
  | { type: 'loadWeights'; resolve: (value: void) => void; reject: (reason?: unknown) => void }
  | { type: 'getAIMove'; resolve: (value: MLResponse) => void; reject: (reason?: unknown) => void }
  | {
      type: 'evaluatePosition';
      resolve: (value: { evaluation: number; status: string; currentPlayer: string }) => void;
      reject: (reason?: unknown) => void;
    }
  | { type: 'getInfo'; resolve: (value: unknown) => void; reject: (reason?: unknown) => void }
  | { type: 'rollDice'; resolve: (value: number) => void; reject: (reason?: unknown) => void };

export class MLAIService {
  private worker: Worker | null = null;
  private initPromise: Promise<void> | null = null;
  private messageCounter = 0;
  private readonly pendingRequests = new Map<number, PendingRequestType>();
  private weightsLoaded = false;
  private networkConfig: {
    value?: { input_size: number; hidden_sizes: number[]; output_size: number };
    policy?: { input_size: number; hidden_sizes: number[]; output_size: number };
  } = {};

  constructor() {
    console.log('ML AI Service: Initializing ML AI Service');
  }

  private init(): Promise<void> {
    if (!this.initPromise) {
      this.initPromise = new Promise((resolve, reject) => {
        if (typeof window === 'undefined') {
          console.log('ML AI Service: Not in browser environment, skipping initialization');
          return resolve();
        }

        console.log('ML AI Service: Creating ML AI Worker');
        this.worker = new Worker(new URL('./ml-ai.worker.ts', import.meta.url), {
          type: 'module',
        });

        const handleMessage = (event: MessageEvent) => {
          console.log('ML AI Service: Received worker message:', event.data.type);

          if (event.data.type === 'ready') {
            console.log('ML AI Service: Worker ready, loading default weights');
            resolve();
          } else if (event.data.type === 'success' || event.data.type === 'error') {
            const promise = this.pendingRequests.get(event.data.id);
            if (promise) {
              if (event.data.type === 'success') {
                console.log('ML AI Service: Request completed successfully:', event.data.id);
                (promise.resolve as (value: unknown) => void)(event.data.response);
              } else {
                console.error('ML AI Service: Request failed:', event.data.id, event.data.error);
                promise.reject(new Error(event.data.error));
              }
              this.pendingRequests.delete(event.data.id);
            }
          }
        };

        const handleError = (error: ErrorEvent) => {
          console.error('ML AI Service: Worker error:', error);
          reject(new Error(`ML AI Worker failed to initialize: ${error.message}`));
        };

        this.worker.onmessage = handleMessage;
        this.worker.onerror = handleError;
      });
    }
    return this.initPromise;
  }

  private async ensureWorkerReady(): Promise<void> {
    console.log('ML AI Service: Ensuring worker is ready...');
    if (!this.initPromise) {
      console.log('ML AI Service: No init promise, starting initialization');
      this.init();
      console.log('ML AI Service: Starting to load default weights');
      await this.loadDefaultWeights();
    }
    console.log('ML AI Service: Waiting for init promise...');
    await this.initPromise;
    if (!this.worker) {
      throw new Error('ML AI Worker not initialized.');
    }
    console.log('ML AI Service: Worker is ready');
  }

  async loadWeights(weights: MLWeights): Promise<void> {
    console.log('ML AI Service: Loading weights...');
    console.log(
      'ML AI Service: Weight sizes - Value:',
      weights.value_weights.length,
      'Policy:',
      weights.policy_weights.length
    );

    if (weights.value_network_config) {
      console.log('ML AI Service: Value network config:', weights.value_network_config);
      this.networkConfig.value = weights.value_network_config;
    }

    if (weights.policy_network_config) {
      console.log('ML AI Service: Policy network config:', weights.policy_network_config);
      this.networkConfig.policy = weights.policy_network_config;
    }

    await this.ensureWorkerReady();
    const messageId = this.messageCounter++;
    const promise = new Promise<void>((resolve, reject) => {
      this.pendingRequests.set(messageId, { type: 'loadWeights', resolve, reject });
    });

    console.log('ML AI Service: Sending weights to worker (id:', messageId, ')');
    this.worker!.postMessage({ id: messageId, type: 'loadWeights', weights });

    await promise;
    this.weightsLoaded = true;
    console.log('ML AI Service: Weights loaded successfully, weightsLoaded flag set to true');
  }

  async getAIMove(gameState: GameState): Promise<MLResponse> {
    console.log('ML AI Service: Getting AI move...');
    console.log('ML AI Service: Current player:', gameState.currentPlayer);
    console.log('ML AI Service: Dice roll:', gameState.diceRoll);
    console.log('ML AI Service: Valid moves:', gameState.validMoves);
    console.log(
      'ML AI Service: Player 1 pieces:',
      gameState.player1Pieces.map(p => ({ square: p.square }))
    );
    console.log(
      'ML AI Service: Player 2 pieces:',
      gameState.player2Pieces.map(p => ({ square: p.square }))
    );

    try {
      const startTime = performance.now();
      await this.ensureWorkerReady();

      console.log('ML AI Service: Worker ready, weights loaded status:', this.weightsLoaded);
      if (!this.weightsLoaded) {
        console.warn('ML AI Service: Weights not loaded, using untrained networks');
      } else {
        console.log('ML AI Service: Using trained neural network weights');
      }

      const messageId = this.messageCounter++;
      const promise = new Promise<MLResponse>((resolve, reject) => {
        this.pendingRequests.set(messageId, { type: 'getAIMove', resolve, reject });
      });

      console.log('ML AI Service: Sending game state to worker (id:', messageId, ')');
      this.worker!.postMessage({ id: messageId, type: 'getAIMove', gameState });

      const response = await promise;
      const totalTime = performance.now() - startTime;

      console.log('ML AI Service: AI move response received in', totalTime.toFixed(2), 'ms');
      console.log('ML AI Service: Chosen move:', response.move);
      console.log('ML AI Service: Position evaluation:', response.evaluation.toFixed(3));
      console.log('ML AI Service: AI thinking:', response.thinking);
      console.log(
        'ML AI Service: Move evaluations:',
        response.diagnostics.move_evaluations.length,
        'moves analyzed'
      );

      return response;
    } catch (error) {
      console.error('ML AI Service: Error getting AI move:', error);

      if (gameState.validMoves.length > 0) {
        const fallbackMove =
          gameState.validMoves[Math.floor(Math.random() * gameState.validMoves.length)];
        console.warn('ML AI Service: Using fallback random move:', fallbackMove);
        return {
          move: fallbackMove,
          evaluation: 0,
          thinking: 'Fallback: Random move due to ML AI error',
          diagnostics: {
            valid_moves: gameState.validMoves,
            move_evaluations: [],
            value_network_output: 0,
            policy_network_outputs: [],
          },
          timings: {
            aiMoveCalculation: 0,
            totalHandlerTime: 0,
          },
        };
      }

      throw new Error(`Failed to get ML AI move: ${error}`);
    }
  }

  async evaluatePosition(
    gameState: GameState
  ): Promise<{ evaluation: number; status: string; currentPlayer: string }> {
    console.log('ML AI Service: Evaluating position...');
    console.log('ML AI Service: Game status:', gameState.gameStatus);
    console.log('ML AI Service: Winner:', gameState.winner);

    await this.ensureWorkerReady();
    const messageId = this.messageCounter++;
    const promise = new Promise<{ evaluation: number; status: string; currentPlayer: string }>(
      (resolve, reject) => {
        this.pendingRequests.set(messageId, { type: 'evaluatePosition', resolve, reject });
      }
    );

    console.log('ML AI Service: Sending position evaluation request (id:', messageId, ')');
    this.worker!.postMessage({ id: messageId, type: 'evaluatePosition', gameState });

    const response = await promise;
    console.log('ML AI Service: Position evaluation result:', response);

    return response;
  }

  async getInfo(): Promise<unknown> {
    console.log('ML AI Service: Getting AI info...');
    await this.ensureWorkerReady();
    const messageId = this.messageCounter++;
    const promise = new Promise<unknown>((resolve, reject) => {
      this.pendingRequests.set(messageId, { type: 'getInfo', resolve, reject });
    });

    console.log('ML AI Service: Sending info request (id:', messageId, ')');
    this.worker!.postMessage({ id: messageId, type: 'getInfo' });

    const response = await promise;
    console.log('ML AI Service: AI info received:', response);

    return response;
  }

  async rollDice(): Promise<number> {
    console.log('ML AI Service: Rolling dice...');
    await this.ensureWorkerReady();
    const messageId = this.messageCounter++;
    const promise = new Promise<number>((resolve, reject) => {
      this.pendingRequests.set(messageId, { type: 'rollDice', resolve, reject });
    });

    console.log('ML AI Service: Sending dice roll request (id:', messageId, ')');
    this.worker!.postMessage({ id: messageId, type: 'rollDice' });

    const result = await promise;
    console.log('ML AI Service: Dice roll result:', result);

    return result;
  }

  private async loadDefaultWeights(): Promise<void> {
    try {
      console.log('ML AI Service: Attempting to load default weights...');

      let response = await fetch('/ml-weights.json.gz');
      if (response.ok) {
        console.log('ML AI Service: Found compressed weights file');
        const compressedData = await response.arrayBuffer();
        console.log('ML AI Service: Compressed data size:', compressedData.byteLength, 'bytes');

        const decompressedData = pako.ungzip(new Uint8Array(compressedData), { to: 'string' });
        console.log(
          'ML AI Service: Decompressed data size:',
          decompressedData.length,
          'characters'
        );

        const weights = JSON.parse(decompressedData) as MLWeights;
        console.log('ML AI Service: Parsed weights, loading into service...');
        await this.loadWeights(weights);
        console.log('ML AI Service: Loaded compressed default weights successfully');
        return;
      }

      console.log('ML AI Service: Compressed weights not found, trying uncompressed...');
      response = await fetch('/ml-weights.json');
      if (response.ok) {
        console.log('ML AI Service: Found uncompressed weights file');
        const weights = (await response.json()) as MLWeights;
        console.log('ML AI Service: Parsed uncompressed weights, loading into service...');
        await this.loadWeights(weights);
        console.log('ML AI Service: Loaded uncompressed default weights successfully');
      } else {
        console.log('ML AI Service: No default weights found, using untrained networks');
      }
    } catch (error) {
      console.error('ML AI Service: Failed to load default weights:', error);
      console.log('ML AI Service: Using untrained networks');
    }
  }
}
