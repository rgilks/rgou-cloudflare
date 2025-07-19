# High Priority TODOs (No Retraining Required)

This document lists critical issues and improvements that can be implemented immediately without retraining the neural network model.

## Critical Bugs (Fix First)

### 1. WASM Weight Persistence Bug

**Status**: ✅ Fixed
**Impact**: AI may be using random weights instead of trained ones

**Problem**:

- `get_ml_ai_move()` creates new `MLAI` instance on every call
- Weights are loaded but instance is discarded immediately
- TypeScript expects persistence but Rust doesn't maintain it

**Solution**:

- ✅ Added global static `MLAI` instance in Rust using `lazy_static`
- ✅ Modified `load_ml_weights()` to store in global instance
- ✅ Modified `get_ml_ai_move()` to use global instance
- ✅ Added error handling for uninitialized AI
- ✅ Added `lazy_static` dependency to Cargo.toml

**Files**: `worker/rust_ai_core/src/wasm_api.rs`, `worker/rust_ai_core/Cargo.toml`

**Implementation**: The fix uses a global `Mutex<Option<MLAI>>` that persists the loaded weights between function calls, ensuring the AI actually uses the trained weights instead of random ones.

### 2. Training Loss Function Issue

**Status**: 🟡 Important
**Impact**: Policy network may not be learning effectively

**Problem**:

- Policy network applies softmax in definition AND uses CrossEntropyLoss
- This can cause incorrect training signals

**Solution**:

- [ ] Remove final softmax from `PolicyNetwork` definition
- [ ] Use raw logits with CrossEntropyLoss
- [ ] Verify training convergence improves
- [ ] Retrain model with corrected loss function

**Files**: `ml/scripts/train_ml_ai.py`

## Performance Improvements

### 3. Enhanced Training Data Generation

**Status**: 🟡 Performance
**Impact**: Faster training data generation, better scalability

**Problem**:

- Per-move subprocess calls create significant overhead
- Thousands of subprocess launches during data generation

**Solution**:

- [ ] Add `simulate_complete_game()` function in Rust
- [ ] Return entire game trajectory in one call
- [ ] Modify Python script to call once per game
- [ ] Reduce inter-process communication overhead

**Files**: `worker/rust_ai_core/src/lib.rs`, `ml/scripts/train_ml_ai.py`

### 4. Better Value Targets

**Status**: 🟡 Quality
**Impact**: More informative training signal

**Problem**:

- Current value target (piece count difference) is crude
- Doesn't capture strategic nuances

**Solution**:

- [ ] Use Classic AI's evaluation function as value targets
- [ ] Scale Classic AI scores appropriately
- [ ] Train value network to mimic Classic AI evaluation
- [ ] Retrain model with better targets

**Files**: `ml/scripts/train_ml_ai.py`

## Code Quality & Documentation

### 5. Add Reproducibility Controls

**Status**: 🟢 Maintainability
**Impact**: Consistent results across training runs

**Solution**:

- [ ] Add random seed control via command line arguments
- [ ] Set seeds for `random`, `numpy`, and `torch`
- [ ] Save training metadata (hyperparameters, loss curves)
- [ ] Document environment requirements

**Files**: `ml/scripts/train_ml_ai.py`

### 6. Continuous Evaluation Script

**Status**: 🟢 Testing
**Impact**: Track AI performance improvements

**Solution**:

- [ ] Create automated testing script
- [ ] Pit ML AI against Classic AI for 100+ games
- [ ] Report win rates and move quality metrics
- [ ] Integrate into development workflow

**Files**: Create new evaluation scripts

### 7. Refactor Training Script

**Status**: 🟢 Maintainability
**Impact**: Better code organization and testability

**Problem**:

- Monolithic 800-line script is hard to maintain

**Solution**:

- [ ] Split into modules: `features.py`, `model.py`, `training.py`
- [ ] Separate data generation, model definition, training logic
- [ ] Improve code organization
- [ ] Add unit tests for individual components

**Files**: `ml/scripts/train_ml_ai.py` → multiple smaller files

## Documentation Updates

### 8. Update Documentation Accuracy

**Status**: 🟢 Documentation
**Impact**: Clear and accurate project documentation

**Solution**:

- [ ] Review all docs for accuracy
- [ ] Remove any remaining incorrect references
- [ ] Update feature descriptions to match implementation
- [ ] Add troubleshooting section

**Files**: All documentation files

## Testing & Validation

### 9. Add Integration Tests

**Status**: 🟢 Testing
**Impact**: Ensure AI components work correctly together

**Solution**:

- [ ] Test WASM weight loading and persistence
- [ ] Test ML AI move selection consistency
- [ ] Test Classic AI vs ML AI performance
- [ ] Add automated regression tests

**Files**: Create new test files

### 10. Performance Benchmarking

**Status**: 🟢 Performance
**Impact**: Ensure AI meets performance requirements

**Solution**:

- [ ] Measure move selection time (target: <10ms)
- [ ] Benchmark memory usage
- [ ] Test with different game states
- [ ] Document performance characteristics

**Files**: Create benchmarking scripts

## Implementation Priority

### Immediate (This Week)

1. Fix WASM weight persistence bug
2. Add reproducibility controls
3. Create continuous evaluation script

### Short-term (Next 2 Weeks)

4. Enhanced training data generation
5. Refactor training script
6. Add integration tests

### Medium-term (Next Month)

7. Better value targets (requires retraining)
8. Performance benchmarking
9. Documentation cleanup

## Success Criteria

- [ ] ML AI consistently uses trained weights
- [ ] Training produces reproducible results
- [ ] Performance meets targets (<10ms move selection)
- [ ] Code is maintainable and well-tested
- [ ] Documentation is accurate and complete

## Notes

- Items marked with "requires retraining" are noted but not included in immediate todos
- Focus on critical bugs first, then performance, then maintainability
- Each improvement should be tested against the Classic AI to measure impact
- Consider the trade-off between complexity and performance gains
