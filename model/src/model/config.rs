use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_shape: (usize, usize, usize), // (channels, height, width)
    pub layers: Vec<LayerConfig>,
    pub routing_iterations: usize,
    pub use_reconstruction: bool,
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
        activation: Activation,
    },
    PrimaryCapsules {
        in_channels: usize,
        capsule_config: CapsuleConfig,
    },
    DigitCapsules {
        input_capsules: usize,
        input_capsule_dim: usize,
        output_capsules: usize,
        output_capsule_dim: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    Tanh,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleConfig {
    pub num_capsules: usize,
    pub capsule_dim: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub validation_split: f32,
    pub save_best: bool,
    pub early_stopping_patience: usize,
    pub optimizer_type: OptimizerType,
    pub loss_config: LossConfig,
    pub lr_schedule: Option<LRSchedule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { momentum: f32 },
    Adam { beta1: f32, beta2: f32, epsilon: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub positive_margin: f32,
    pub negative_margin: f32,
    pub down_weighting: f32,
    pub reconstruction_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRSchedule {
    StepDecay { step_size: usize, gamma: f32 },
    ExponentialDecay { gamma: f32 },
    ReduceOnPlateau { factor: f32, patience: usize },
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_shape: (3, 64, 64),
            layers: vec![
                LayerConfig::Conv2d {
                    in_channels: 3,
                    out_channels: 64,
                    kernel_size: 3,
                    stride: 1,
                    padding: 1,
                    activation: Activation::ReLU,
                },
                LayerConfig::Conv2d {
                    in_channels: 64,
                    out_channels: 128,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1,
                    activation: Activation::ReLU,
                },
                LayerConfig::PrimaryCapsules {
                    in_channels: 128,
                    capsule_config: CapsuleConfig {
                        num_capsules: 32,
                        capsule_dim: 8,
                        kernel_size: 9,
                        stride: 2,
                        padding: 0,
                    },
                },
                LayerConfig::DigitCapsules {
                    input_capsules: 32,
                    input_capsule_dim: 8,
                    output_capsules: 2,
                    output_capsule_dim: 16,
                },
            ],
            routing_iterations: 3,
            use_reconstruction: false,
            extra_params: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            learning_rate: 0.001,
            num_epochs: 30,
            validation_split: 0.2,
            save_best: true,
            early_stopping_patience: 5,
            optimizer_type: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            loss_config: LossConfig {
                positive_margin: 0.9,
                negative_margin: 0.1,
                down_weighting: 0.5,
                reconstruction_weight: 0.0005,
            },
            lr_schedule: Some(LRSchedule::ReduceOnPlateau {
                factor: 0.5,
                patience: 3,
            }),
        }
    }
}

// Validation de configuration
impl NetworkConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("Le réseau doit avoir au moins une couche".to_string());
        }
        
        if self.routing_iterations == 0 {
            return Err("routing_iterations doit être > 0".to_string());
        }

        // Vérifier la cohérence des dimensions
        let mut current_channels = self.input_shape.0;
        
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerConfig::Conv2d { in_channels, out_channels, .. } => {
                    if *in_channels != current_channels {
                        return Err(format!(
                            "Couche {}: in_channels ({}) ne correspond pas à la sortie précédente ({})",
                            i, in_channels, current_channels
                        ));
                    }
                    current_channels = *out_channels;
                }
                LayerConfig::PrimaryCapsules { in_channels, .. } => {
                    if *in_channels != current_channels {
                        return Err(format!(
                            "Couche {}: in_channels ({}) ne correspond pas à la sortie précédente ({})",
                            i, in_channels, current_channels
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("batch_size doit être > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate doit être > 0".to_string());
        }
        if self.validation_split < 0.0 || self.validation_split >= 1.0 {
            return Err("validation_split doit être dans [0, 1)".to_string());
        }
        Ok(())
    }
}
