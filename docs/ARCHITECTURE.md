# Architecture Overview

This document details the architecture of the Royal Game of Ur project, focusing on its AI engine, frontend, deployment, and infrastructure.

## What Makes This Special?

This implementation stands out for several reasons:

- **Ancient Game, Modern Tech**: Brings a 4500-year-old game to life with cutting-edge web technologies
- **Dual AI System**: Features both a classic expectiminimax AI and a neural network AI, each with distinct playstyles
- **Browser-Native AI**: All AI runs locally in your browser via WebAssembly - no server calls needed
- **Offline-First**: Works completely offline once loaded, perfect for mobile or unreliable connections
- **Performance**: Rust-compiled AI provides desktop-level performance in the browser
- **Evolutionary Architecture**: Successfully migrated from hybrid client/server AI to pure client-side execution

## Core Principles

- **High Performance**: Rust and WebAssembly for AI
- **Offline Capability**: Fully playable without internet
- **Seamless UX**: Modern, responsive UI
- **Maintainability**: Clear separation of UI, logic, and AI

## System Architecture

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
- **Architecture**: Pure client-side execution via Web Workers

### WASM Architecture Evolution

The project has evolved from a hybrid client/server architecture to a pure client-side implementation:

**Original Design (Early Development)**:

- AI computation could run on either client (WASM) or server (Cloudflare Worker)
- Server-side AI provided backup and potential performance benefits
- More complex deployment and infrastructure requirements

**Current Design (Production)**:

- All AI computation runs client-side via WebAssembly workers
- Eliminates network latency and server infrastructure costs
- Enables true offline play without server dependencies
- Simplified deployment and reduced attack surface

**Preserved Infrastructure**:

- Cloudflare Worker code remains in `worker/src/lib.rs` for potential future use
- Server-side AI endpoints (`/ai-move`, `/health`) are inactive but available
- Architecture supports easy reactivation if server-side features are needed

## Data Flow

### AI Turn Processing

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

## Database System

### Local Development

- **Database**: SQLite (`local.db`)
- **ORM**: Drizzle ORM
- **Setup**: `npm run db:local:reset`

### Production

- **Database**: Cloudflare D1
- **ORM**: Drizzle ORM
- **Migrations**: `npm run migrate:d1`

### Schema

```typescript
// src/lib/db/schema.ts
export const games = sqliteTable('games', {
  id: text('id')
    .primaryKey()
    .$defaultFn(() => nanoid()),
  playerId: text('playerId').notNull(),
  winner: text('winner', { enum: ['player1', 'player2'] }),
  completedAt: integer('completedAt', { mode: 'timestamp_ms' }),
  moveCount: integer('moveCount'),
  duration: integer('duration'),
  history: text('history', { mode: 'json' }),
  gameType: text('gameType', { enum: ['classic', 'ml', 'watch'] }),
});
```

### Game Statistics

The game includes comprehensive statistics tracking:

**Features**:

- **Win/Loss Tracking**: Automatic recording of game outcomes
- **Win Rate Calculation**: Percentage of games won
- **Local Storage**: Statistics persist across browser sessions
- **Database Integration**: Games saved to database for analytics
- **Real-time Updates**: Statistics update immediately after game completion

**Implementation**:

- Statistics managed using Zustand with persistent storage
- Games automatically saved to database upon completion
- Privacy-focused: Player ID generated using `nanoid()` for anonymous tracking

## Deployment

### Frontend Deployment

- **Platform**: Next.js on Cloudflare Pages
- **Build**: `npm run build:cf`
- **Domain**: `https://rgou.tre.systems`

### WASM Security Headers

Set in `public/_headers`:

```
/wasm/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Resource-Policy: same-origin
```

## Preserved Server Infrastructure

### Cloudflare Worker (`worker/src/lib.rs`)

**Status**: Inactive (preserved for future use)

**Available Endpoints**:

- `POST /ai-move` - Server-side AI computation
- `GET /health` - Health check endpoint

**Potential Future Use Cases**:

1. **Server-Side Validation**: Validate client moves in multiplayer
2. **Analytics**: Collect AI performance metrics
3. **Multiplayer Support**: Centralized AI for unreliable clients
4. **AI Model Distribution**: Dynamic AI selection based on device capabilities

**Reactivation**:

```bash
cd worker
wrangler deploy
```

### Performance Considerations

**Server-Side AI Advantages**:

- Consistent performance across devices
- Reduced client load
- Centralized control

**Server-Side AI Disadvantages**:

- Network latency
- Infrastructure costs
- Offline limitations
- Scalability concerns

**Current Decision**: Client-side AI exclusively for performance, offline capability, and cost efficiency.

## Development vs Production

### Development Environment

- **Dev-only tools**: AI diagnostics, AI toggle, reset/test buttons (only on localhost)
- **Local database**: SQLite for development
- **Debug features**: Enhanced logging and diagnostics

### Production Environment

- **Clean UI**: No development tools
- **Classic AI default**: Most reliable AI opponent
- **Cloudflare D1**: Production database
- **Optimized builds**: Minified and optimized assets

## Summary

- Modern, maintainable, high-performance architecture
- All AI runs locally in the browser (WASM)
- Clear separation of concerns
- Full offline and online support
- Comprehensive statistics tracking
- Privacy-focused data collection
- Preserved server infrastructure for future use cases
