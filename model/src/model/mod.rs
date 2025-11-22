pub mod config;
pub mod layers;
pub mod capsule;
pub mod routing;
pub mod optimizer;
pub mod loss;
pub mod builder;
pub mod core;

// Réexportations principales
pub use config::{NetworkConfig, TrainingConfig};
pub use core::CapNet;
pub use builder::ModelBuilder;
pub use optimizer::{Optimizer, Adam, SGD};
pub use loss::{MarginLoss, LossFunction};
