use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_shape: (usize, usize, usize),
    pub layers: Vec<LayerConfig>,
    pub routing_iterations: usize,
    pub extra_params: Option<HashMap<String, f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerConfig {
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Option<String>,
    },
    PrimaryCapsules {
        in_channels: usize,
        capsule_config: CapsuleConfig,
    },
    DigitCapsules {
        primary_capsules: usize,
        primary_capsule_dim: usize,
        capsule_config: CapsuleConfig,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleConfig {
    pub num_capsules: usize,
    pub capsule_dim: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,  // NOUVEAU: padding pour les capsules
    pub capsule_params: Option<HashMap<String, f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub validation_split: f32,
    pub save_best: bool,
    pub early_stopping_patience: usize,
    pub loss_function: String,
    pub optimizer: String,
    pub margin_loss_params: Option<MarginLossConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginLossConfig {
    pub positive_margin: f32,
    pub negative_margin: f32,
    pub down_weighting: f32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_shape: (3, 64, 64),
            layers: vec![
                LayerConfig::Conv2d {
                    in_channels: 3,
                    out_channels: 32,
                    kernel_size: 3,
                    stride: 1,
                    padding: 1,
                    activation: Some("relu".to_string()),
                },
                LayerConfig::PrimaryCapsules {
                    in_channels: 32,
                    capsule_config: CapsuleConfig {
                        num_capsules: 8,
                        capsule_dim: 4,
                        kernel_size: 3,
                        stride: 2,
                        padding: 1,  // PADDING AJOUTÉ
                        capsule_params: None,
                    },
                },
                LayerConfig::DigitCapsules {
                    primary_capsules: 8,
                    primary_capsule_dim: 4,
                    capsule_config: CapsuleConfig {
                        num_capsules: 2,
                        capsule_dim: 8,
                        kernel_size: 0,
                        stride: 0,
                        padding: 0,
                        capsule_params: None,
                    },
                },
            ],
            routing_iterations: 3,
            extra_params: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            num_epochs: 50,
            validation_split: 0.2,
            save_best: true,
            early_stopping_patience: 10,
            loss_function: "margin".to_string(),
            optimizer: "adam".to_string(),
            margin_loss_params: Some(MarginLossConfig {
                positive_margin: 0.9,
                negative_margin: 0.1,
                down_weighting: 0.5,
            }),
        }
    }
}