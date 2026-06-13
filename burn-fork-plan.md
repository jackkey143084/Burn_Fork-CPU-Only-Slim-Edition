# Burn Fork Plan — CPU-Only Slim Edition

**Goal:** Fork [tracel-ai/burn](https://github.com/tracel-ai/burn) and cut the codebase roughly in half by keeping only the CPU training path and removing all GPU backends, distributed infrastructure, and non-essential extras.

**Target:** ~1,000 files removed from 2,012 → estimated **~900–1,000 files remaining**

---

## What We Keep

| Crate | Files | Purpose |
|---|---|---|
| `burn` | 5 | Top-level facade crate |
| `burn-std` | 30 | Core primitives, no_std compat |
| `burn-backend` | 37 | Backend trait definitions |
| `burn-ir` | 9 | Intermediate representation |
| `burn-derive` | 32 | Procedural macros (`#[derive(Module)]`) |
| `burn-tensor` | 66 | Tensor API |
| `burn-autodiff` | 39 | Automatic differentiation |
| `burn-core` | 49 | Module system, record, config |
| `burn-nn` | 91 | Neural network layers |
| `burn-optim` | 35 | Optimizers (SGD, Adam, etc.) |
| `burn-ndarray` | 42 | **The one backend** — pure CPU via ndarray |
| `burn-fusion` | 50 | Op fusion layer (benefits ndarray perf) |
| `burn-flex` | 49 | Flexible dispatch abstraction |
| `burn-dispatch` | 18 | Backend dispatch (stripped to ndarray only) |
| `burn-train` | 147 | Training loop, learner, metrics |
| `burn-store` | 172 | Model serialization (safetensors, burnpack) |
| `burn-dataset` | 60 | Dataset loading utilities |
| **Total** | **~931** | |

---

## What Gets Removed

### GPU / Accelerator Backends (~200 files)

| Crate | Files | Why removed |
|---|---|---|
| `burn-wgpu` | 5 | WebGPU backend |
| `burn-cuda` | 3 | CUDA backend |
| `burn-rocm` | 3 | ROCm/AMD GPU backend |
| `burn-tch` | 23 | LibTorch/PyTorch backend |
| `burn-candle` | 19 | Candle (HuggingFace) backend |
| `burn-cpu` | 3 | CubeCL-based CPU backend (redundant with ndarray) |
| `burn-cubecl` | 112 | CubeCL GPU kernel library |
| `burn-cubecl-fusion` | 55 | CubeCL fusion (GPU only) |

**Remove the `cubecl` / `cubecl-common` / `cubecl-zspace` / `cubek` git dependencies from workspace `Cargo.toml`** — these are GPU compute graph libs with no CPU path.

### Distributed / Remote Infrastructure (~58 files)

| Crate | Files | Why removed |
|---|---|---|
| `burn-remote` | 24 | HTTP/WebSocket remote backend |
| `burn-router` | 23 | Multi-device routing |
| `burn-communication` | 11 | Serialized tensor comms |

### Domain / Optional Extras (~75 files)

| Crate | Files | Why removed |
|---|---|---|
| `burn-rl` | 13 | Reinforcement learning utilities |
| `burn-vision` | 63 | Computer vision ops |
| `burn-backend-extension` | 5 | Extension traits (only used by GPU crates) |

### Test Infrastructure (~446 files)

| Crate | Files | Why removed |
|---|---|---|
| `burn-backend-tests` | 435 | Cross-backend test suite (only needed when you have multiple backends) |
| `burn-no-std-tests` | 11 | no_std integration tests |

### Top-Level Non-Code (~231 files)

| Directory | Files | Why removed |
|---|---|---|
| `examples/` | 152 | All examples (depend on removed backends) |
| `burn-book/` | 46 | mdBook documentation source |
| `contributor-book/` | 24 | Contributor guide mdBook |
| `xtask/` | 9 | Build automation scripts (CI-specific) |

**Total removed: ~1,010 files**

---

## Step-by-Step Fork Instructions

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/burn.git burn-slim
cd burn-slim
```

### 2. Remove Top-Level Non-Code Directories

```bash
rm -rf examples/ burn-book/ contributor-book/ xtask/ docs/ assets/
rm -f benchmarks.toml deny.toml codecov.yml _typos.toml POEM.md CITATION.cff
```

### 3. Remove Crate Directories

```bash
cd crates
rm -rf burn-wgpu burn-cuda burn-rocm burn-tch burn-candle burn-cpu
rm -rf burn-cubecl burn-cubecl-fusion
rm -rf burn-remote burn-router burn-communication
rm -rf burn-rl burn-vision burn-backend-extension
rm -rf burn-backend-tests burn-no-std-tests
```

### 4. Rewrite Root `Cargo.toml` Workspace

Replace the `[workspace]` section:

```toml
[workspace]
resolver = "2"

members = [
    "crates/burn",
    "crates/burn-std",
    "crates/burn-backend",
    "crates/burn-ir",
    "crates/burn-derive",
    "crates/burn-tensor",
    "crates/burn-autodiff",
    "crates/burn-core",
    "crates/burn-nn",
    "crates/burn-optim",
    "crates/burn-ndarray",
    "crates/burn-fusion",
    "crates/burn-flex",
    "crates/burn-dispatch",
    "crates/burn-train",
    "crates/burn-store",
    "crates/burn-dataset",
    "crates/burn-store/pytorch-tests",
    "crates/burn-store/safetensors-tests",
]
```

Remove from `[workspace.dependencies]`:
- `candle-core`
- `tch`, `torch-sys`
- `nvml-wrapper` (GPU memory monitoring — optional, used in train metrics)
- `axum`, `tokio-tungstenite` (remote backend only)
- `cubecl`, `cubecl-common`, `cubecl-zspace`, `cubek` (all git deps)
- `opentelemetry`, `opentelemetry-aws`, `opentelemetry-otlp`, `opentelemetry_sdk`, `tracing-opentelemetry` (used only in remote/router)
- `text_placeholder` (WGPU shader templating)

Remove from `[workspace.dependencies]` internal crates section:
- `burn-candle`, `burn-communication`, `burn-cubecl`, `burn-cubecl-fusion`
- `burn-cuda`, `burn-cpu`, `burn-rocm`, `burn-remote`, `burn-rocm`
- `burn-router`, `burn-rl`, `burn-tch`, `burn-vision`, `burn-wgpu`
- `burn-backend-extension`, `burn-backend-tests`, `burn-no-std-tests`

### 5. Update `crates/burn/Cargo.toml`

Remove optional dependencies for all removed crates:
```toml
# REMOVE these lines:
burn-candle = ...
burn-cuda = ...
burn-cpu = ...
burn-rocm = ...
burn-flex = ...   # keep only if used by train path
burn-router = ...
burn-tch = ...
burn-wgpu = ...
burn-rl = ...
burn-vision = ...
```

Remove matching `[features]` entries that gate them.

Keep:
```toml
[dependencies]
burn-core    = { workspace = true }
burn-train   = { workspace = true, optional = true }
burn-store   = { workspace = true, optional = true }
burn-nn      = { workspace = true }
burn-optim   = { workspace = true, optional = true }
burn-std     = { workspace = true }
burn-ndarray = { workspace = true, optional = true }
burn-autodiff = { workspace = true, optional = true }
burn-dispatch = { workspace = true, optional = true }

[features]
default = ["ndarray"]
ndarray = ["burn-ndarray"]
train   = ["burn-train", "burn-store"]
autodiff = ["burn-autodiff"]
```

### 6. Update `crates/burn-dispatch/Cargo.toml`

Strip all GPU backend feature flags — leave only ndarray:

```toml
[dependencies]
burn-backend  = { workspace = true }
burn-autodiff = { workspace = true, optional = true }
burn-flex     = { workspace = true }
burn-ndarray  = { workspace = true, optional = true }
paste         = { workspace = true }

[features]
default  = ["ndarray"]
ndarray  = ["burn-ndarray"]
autodiff = ["burn-autodiff"]
```

Remove the macro invocations in `dispatch.rs` that reference `wgpu`, `cuda`, `tch`, `candle`, `rocm`, `remote`.

### 7. Update `crates/burn-train/Cargo.toml`

Remove the `nvml-wrapper`, `sysinfo`, `systemstat` optional dependencies if desired (GPU/system metrics). They compile fine without a GPU but add weight — mark as truly optional and disable in default features.

Remove `burn-rl` optional dep.

### 8. Update `crates/burn-std/Cargo.toml`

Remove the `cubecl-common` and `cubecl-zspace` git dependencies:

```toml
# REMOVE:
cubecl-common = { workspace = true, ... }
cubecl-zspace = { workspace = true, ... }
```

Then audit `crates/burn-std/src/` for any `cubecl` imports and replace with pure-Rust equivalents (mainly the `FloatElement`/`IntElement` type registry — these have a thin CubeCL dependency that needs to be inlined or replaced).

> **Note:** `burn-std` depending on CubeCL is the trickiest part of the fork. CubeCL types (`Elem`, `Feature`) leak into the backend trait layer. Two options:
> - **Option A (easier):** Keep the git dep for `cubecl-common` only (it's small, pure Rust, no GPU code). This keeps the fork simpler at the cost of one git dep.
> - **Option B (cleaner):** Copy the 3-4 needed types from cubecl-common into a `burn-std::elem` module and remove the dep entirely.

### 9. Update `crates/burn-backend/Cargo.toml`

Remove optional `cubecl` dep. Audit for `#[cfg(feature = "cubecl")]` gated code and remove.

### 10. Verify the Build

```bash
# Check workspace resolves
cargo metadata --no-deps 2>&1 | head -20

# Build the core stack
cargo build -p burn-tensor
cargo build -p burn-autodiff
cargo build -p burn-ndarray
cargo build -p burn-nn
cargo build -p burn-optim
cargo build -p burn-train

# Full workspace check
cargo check --workspace
```

### 11. Run Remaining Tests

```bash
# Core tensor tests
cargo test -p burn-tensor

# Autodiff tests  
cargo test -p burn-autodiff

# NdArray backend tests
cargo test -p burn-ndarray

# NN layer tests
cargo test -p burn-nn

# Store tests
cargo test -p burn-store
```

---

## Estimated File Count After Fork

| Category | Before | After |
|---|---|---|
| `crates/` | 1,739 | ~731 |
| `examples/` | 152 | 0 |
| `burn-book/` | 46 | 0 |
| `contributor-book/` | 24 | 0 |
| `xtask/` | 9 | 0 |
| Root files | 18 | ~12 |
| `.github/` | 14 | ~4 (keep basic CI) |
| **Total** | **2,012** | **~747** |

That's roughly a **63% reduction** — well past the "halve it" goal.

---

## What Still Works After the Fork

✅ Define models with `#[derive(Module)]`  
✅ Tensor operations (all dtypes, all shapes)  
✅ Automatic differentiation  
✅ All NN layers: Linear, Conv2d, LSTM, Attention, BatchNorm, Dropout, etc.  
✅ All optimizers: SGD, Adam, AdamW, RMSprop, etc.  
✅ Training loop with metrics and checkpointing  
✅ Model serialization: safetensors, burnpack formats  
✅ Dataset loading: CSV, SQLite, image, audio  
✅ Op fusion for ndarray (via burn-fusion)  

❌ GPU acceleration (WGPU, CUDA, ROCm, Metal)  
❌ LibTorch/PyTorch backend  
❌ Candle backend  
❌ Distributed/multi-device training  
❌ Reinforcement learning helpers  
❌ Computer vision ops (burn-vision)  
❌ ONNX model import  

---

## Key Risk: `burn-std` → `cubecl-common` Dependency

This is the only non-trivial coupling. `burn-std` uses `cubecl-common` for:
- `FloatElement` / `IntElement` dtype traits
- `Feature` enum (backend capability flags)

If you want zero CubeCL deps, copy ~150 lines from `cubecl-common/src/elem.rs` into `burn-std/src/elem.rs` and update the re-exports. Otherwise, keep the git dep — it's pure Rust with no GPU code.

---

## Suggested Crate Name for the Fork

To avoid Cargo registry conflicts if you ever publish:

```toml
# In each crate's Cargo.toml, rename:
burn          → burn-slim
burn-ndarray  → burn-slim-ndarray
# etc.
```

Or keep the `burn-*` names and just don't publish to crates.io — use path deps internally.
