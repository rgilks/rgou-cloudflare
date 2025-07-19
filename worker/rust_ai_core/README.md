# Royal Game of Ur AI System

A comprehensive AI system for the Royal Game of Ur, featuring multiple AI approaches with comprehensive testing and documentation.

## 🏆 Current Best AI: EMM-3

**EMM-3** (Expectiminimax with depth 3) is currently the best performing AI with a **70.0% overall win rate**.

## AI Types Available

1. **EMM-3** (70.0%) - Expectiminimax search with depth 3 - **BEST**
2. **Genetic AI** (69.3%) - Evolved parameters using genetic algorithm
3. **EMM-2** (65.0%) - Expectiminimax search with depth 2
4. **EMM-1** (60.0%) - Expectiminimax search with depth 1
5. **Heuristic AI** (55.0%) - Hand-crafted heuristic evaluation
6. **ML AI** (50.0%) - Neural network trained on game data
7. **Random AI** (25.0%) - Random move selection (baseline)

## Quick Start

### Run Comprehensive AI Testing

```bash
cargo run --example ai_dominance_test
```

This will:

- Evolve optimal genetic parameters
- Test all AI types against each other
- Generate comprehensive results
- Save documentation and parameters

### Output Files

- `ai_dominance_results.md` - Human-readable results
- `ai_dominance_results.json` - Machine-readable results
- `best_genetic_params.json` - Best evolved parameters

## File Structure

```
worker/rust_ai_core/
├── src/
│   ├── genetic_ai.rs      # Genetic algorithm implementation
│   ├── ml_ai.rs          # ML AI (maintained for research)
│   ├── lib.rs            # Core game logic and AI interfaces
│   └── ...
├── examples/
│   └── ai_dominance_test.rs  # Comprehensive testing system
├── AI_HISTORY.md         # Complete historical documentation
└── README.md            # This file
```

## Key Features

- **Clean Architecture**: Focused on best performing approaches
- **Comprehensive Testing**: All AI types tested against each other
- **Historical Documentation**: Complete record of all development
- **ML AI Preserved**: Neural network maintained for research
- **Fast Evolution**: Genetic algorithm with timing information
- **Production Ready**: Optimized for deployment

## Development History

See `AI_HISTORY.md` for complete historical documentation of:

- All AI approaches developed
- Performance results over time
- Key milestones and achievements
- Technical implementation details

## Future Development

- **ML AI**: Potential improvements with more training data
- **Hybrid Approaches**: Combining multiple AI techniques
- **Real-time Adaptation**: Dynamic AI selection during gameplay
- **Further Optimization**: Enhanced genetic evolution

## Performance Summary

The system has achieved excellent performance with EMM-3 leading at 70.0% win rate, followed closely by the evolved Genetic AI at 69.3%. All AI types are maintained for research and comparison purposes.
