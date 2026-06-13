What We Keep
Crate	Files	Purpose
burn	5	Top-level facade crate
burn-std	30	Core primitives, no_std compat
burn-backend	37	Backend trait definitions
burn-ir	9	Intermediate representation
burn-derive	32	Procedural macros (#[derive(Module)])
burn-tensor	66	Tensor API
burn-autodiff	39	Automatic differentiation
burn-core	49	Module system, record, config
burn-nn	91	Neural network layers
burn-optim	35	Optimizers (SGD, Adam, etc.)
burn-ndarray	42	The one backend — pure CPU via ndarray
burn-fusion	50	Op fusion layer (benefits ndarray perf)
burn-flex	49	Flexible dispatch abstraction
burn-dispatch	18	Backend dispatch (stripped to ndarray only)
burn-train	147	Training loop, learner, metrics
burn-store	172	Model serialization (safetensors, burnpack)
burn-dataset	60	Dataset loading utilities
