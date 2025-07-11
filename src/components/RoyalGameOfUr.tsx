'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useGameStore, useGameState, useGameActions } from '@/lib/game-store';
import { soundEffects } from '@/lib/sound-effects';
import { isLocalDevelopment } from '@/lib/utils';
import GameBoard from './GameBoard';
import AnimatedBackground from './AnimatedBackground';
import { motion } from 'framer-motion';
import { Sparkles, ExternalLink } from 'lucide-react';
import Image from 'next/image';
import AIDiagnosticsPanel from './AIDiagnosticsPanel';
import HowToPlayPanel from './HowToPlayPanel';

export default function RoyalGameOfUr() {
  const gameState = useGameState();
  const { processDiceRoll, makeMove, makeAIMove, reset } = useGameActions();
  const aiThinking = useGameStore(state => state.aiThinking);
  const lastAIDiagnostics = useGameStore(state => state.lastAIDiagnostics);
  const lastAIMoveDuration = useGameStore(state => state.lastAIMoveDuration);
  const lastMoveType = useGameStore(state => state.lastMoveType);
  const lastMovePlayer = useGameStore(state => state.lastMovePlayer);

  const [aiSource, setAiSource] = useState<'server' | 'client'>('client');
  const [soundEnabled, setSoundEnabled] = useState(true);

  const [diagnosticsPanelOpen, setDiagnosticsPanelOpen] = useState(false);
  const [howToPlayOpen, setHowToPlayOpen] = useState(false);

  useEffect(() => {
    if (
      gameState.currentPlayer === 'player2' &&
      !gameState.canMove &&
      gameState.gameStatus === 'playing'
    ) {
      setTimeout(() => processDiceRoll(), 500);
    }
  }, [gameState.currentPlayer, gameState.canMove, gameState.gameStatus, processDiceRoll]);

  useEffect(() => {
    if (gameState.currentPlayer === 'player2' && gameState.canMove) {
      soundEffects.aiThinking();
      makeAIMove(aiSource);
    }
  }, [gameState, makeAIMove, aiSource]);

  useEffect(() => {
    if (gameState.gameStatus === 'finished') {
      setTimeout(() => {
        if (gameState.winner === 'player1') {
          soundEffects.gameWin();
        } else {
          soundEffects.gameLoss();
        }
      }, 500);
    }
  }, [gameState.gameStatus, gameState.winner]);

  useEffect(() => {
    if (lastMoveType && lastMovePlayer) {
      switch (lastMoveType) {
        case 'capture':
          soundEffects.pieceCapture();
          break;
        case 'rosette':
          soundEffects.rosetteLanding();
          break;
        case 'finish':
          soundEffects.pieceFinish();
          break;
        case 'move':
          soundEffects.pieceMove();
          break;
      }
    }
  }, [lastMoveType, lastMovePlayer]);

  useEffect(() => {
    if (
      gameState.currentPlayer === 'player1' &&
      !gameState.canMove &&
      gameState.diceRoll === null &&
      gameState.gameStatus === 'playing'
    ) {
      setTimeout(() => processDiceRoll(), 500);
    }
  }, [
    gameState.currentPlayer,
    gameState.canMove,
    gameState.diceRoll,
    gameState.gameStatus,
    processDiceRoll,
  ]);

  const handlePieceClick = useCallback(
    (pieceIndex: number) => {
      if (
        gameState.canMove &&
        gameState.validMoves.includes(pieceIndex) &&
        gameState.currentPlayer === 'player1'
      ) {
        makeMove(pieceIndex);
      }
    },
    [gameState.canMove, gameState.validMoves, gameState.currentPlayer, makeMove]
  );

  const handleReset = () => {
    reset();
  };

  const toggleSound = () => {
    const newState = soundEffects.toggle();
    setSoundEnabled(newState);
  };

  const showHowToPlay = () => {
    setHowToPlayOpen(true);
  };

  const diagnosticsPanel =
    isLocalDevelopment() && lastAIDiagnostics ? (
      <AIDiagnosticsPanel
        lastAIDiagnostics={lastAIDiagnostics}
        lastAIMoveDuration={lastAIMoveDuration}
        isOpen={diagnosticsPanelOpen}
        onToggle={() => setDiagnosticsPanelOpen(!diagnosticsPanelOpen)}
        gameState={gameState}
      />
    ) : null;

  return (
    <>
      <AnimatedBackground />
      <div className="relative min-h-screen w-full flex items-center justify-center p-4">
        {/* Pop Out Button - Desktop only */}
        <div className="hidden md:block absolute top-4 right-4 z-50">
          <button
            onClick={() => {
              window.open(
                '/',
                'GamePopout',
                'width=420,height=800,menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=no'
              );
            }}
            className="glass-dark rounded-lg px-4 py-2 flex items-center space-x-2 text-white/80 hover:text-white font-semibold shadow-lg backdrop-blur-md border border-white/10 transition-colors"
            title="Pop Out Game"
          >
            <ExternalLink className="w-4 h-4 mr-1" />
            <span>Pop Out Game</span>
          </button>
        </div>
        {isLocalDevelopment() && (
          <div className="hidden xl:block absolute left-4 top-1/2 -translate-y-1/2 w-80">
            {diagnosticsPanel}
          </div>
        )}

        <motion.div
          className="w-full max-w-sm mx-auto space-y-3"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div
            className="text-center space-y-1"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            <motion.h1
              className="text-2xl md:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-400 neon-text"
              animate={{
                backgroundPosition: ['0%', '100%', '0%'],
              }}
              transition={{
                duration: 8,
                repeat: Infinity,
                ease: 'linear',
              }}
              style={{
                backgroundSize: '200% 200%',
              }}
            >
              Royal Game of Ur
            </motion.h1>

            <motion.div
              className="flex items-center justify-center space-x-2"
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ repeat: Infinity, duration: 2.5 }}
            >
              <Sparkles className="w-3 h-3 text-amber-400" />
              <span className="text-white/80 font-medium text-sm">
                Ancient Mesopotamian Board Game
              </span>
              <Sparkles className="w-3 h-3 text-amber-400" />
            </motion.div>
          </motion.div>

          <GameBoard
            gameState={gameState}
            onPieceClick={handlePieceClick}
            aiThinking={aiThinking}
            onResetGame={handleReset}
            aiSource={aiSource}
            onAiSourceChange={setAiSource}
            soundEnabled={soundEnabled}
            onToggleSound={toggleSound}
            onShowHowToPlay={showHowToPlay}
          />

          {isLocalDevelopment() && <div className="xl:hidden">{diagnosticsPanel}</div>}

          <HowToPlayPanel isOpen={howToPlayOpen} onClose={() => setHowToPlayOpen(false)} />

          <motion.div
            className="text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1, duration: 0.6 }}
          >
            <p className="mt-4 text-center">
              <a href="https://ko-fi.com/N4N31DPNUS" target="_blank" rel="noopener noreferrer">
                <Image
                  width={145}
                  height={36}
                  className="block mx-auto"
                  src="https://storage.ko-fi.com/cdn/kofi2.png?v=6"
                  alt="Buy Me a Coffee at ko-fi.com"
                />
              </a>
            </p>
          </motion.div>
        </motion.div>
      </div>
    </>
  );
}
