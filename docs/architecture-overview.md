# Architecture Overview

This document details the architecture of the Royal Game of Ur project, focusing on its AI engine, frontend, and deployment.

## What Makes This Special?

This implementation stands out for several reasons:

- **Ancient Game, Modern Tech**: Brings a 4500-year-old game to life with cutting-edge web technologies
- **Dual AI System**: Features both a classic expectiminimax AI and a neural network AI, each with distinct playstyles
- **Browser-Native AI**: All AI runs locally in your browser via WebAssembly - no server calls needed
- **Offline-First**: Works completely offline once loaded, perfect for mobile or unreliable connections
- **Performance**: Rust-compiled AI provides desktop-level performance in the browser

## Principles

- High performance: Rust and WebAssembly for AI
- Offline capability: Fully playable without internet
- Seamless UX: Modern, responsive UI
- Maintainability: Clear separation of UI, logic, and AI

## Core Components

1. **Next.js Frontend**: React app for UI and game state
2. **AI (WebAssembly)**: Rust AI logic compiled to Wasm for browser (Classic AI and ML AI)
3. **Database**: Cloudflare D1, Drizzle ORM

The shared Rust AI core (`worker/rust_ai_core`) contains all game rules, evaluation, and expectiminimax search. Both Classic AI and ML AI use this for identical strategy, running locally in the browser.

For AI algorithm details, see [AI System](./ai-system.md). For ML AI, see [ML AI System](./ml-ai-system.md).

### Frontend (`src/`)

- **UI Components**: `src/components/` (React, Tailwind, Framer Motion)
- **State Management**: `src/lib/game-store.ts` (Zustand + Immer)
- **Game Logic**: `src/lib/game-logic.ts` (pure functions)
- **AI Services**: `src/lib/wasm-ai-service.ts` (Classic AI), `src/lib/ml-ai-service.ts` (ML AI)
- **Database**: `src/lib/actions.ts` (save games)
- **Statistics**: `src/lib/stats-store.ts`

### AI Engine

- **Classic AI**: Rust, expectiminimax, compiled to WebAssembly
- **ML AI**: Rust, neural network, compiled to WebAssembly
- **Performance**: All AI runs locally in the browser (no server calls)

### Data Flow: AI Turn

1. `RoyalGameOfUr.tsx` detects AI turn
2. Calls `makeAIMove` in `game-store.ts`
3. Calls appropriate AI service (Classic AI or ML AI)
4. Chosen move processed by `makeMoveLogic`
5. UI updates

### Game Completion & Database

1. Game state set to finished
2. Local stats updated
3. `postGameToServer` action runs
4. Game saved to DB
5. Completion overlay shows stats

## Database

- **Local**: SQLite (`local.db`), Drizzle ORM
- **Production**: Cloudflare D1, Drizzle ORM
- **Schema**: See `src/lib/db/schema.ts`

## Deployment

- **Frontend**: Next.js on Cloudflare Pages
- **Database**: Cloudflare D1
- **Automation**: GitHub Actions workflow

### WASM Security Headers

Set in `public/_headers`:

```
/wasm/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Resource-Policy: same-origin
```

## Development vs Production UI

- **Dev-only tools**: AI diagnostics, AI toggle, reset/test buttons (only on localhost)
- **Production**: Clean UI, Classic AI default

## Summary

- Modern, maintainable, high-performance architecture
- All AI runs locally in the browser (WASM)
- Clear separation of concerns
- Full offline and online support
