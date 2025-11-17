use super::{
    config::{NetworkConfig, TrainingConfig, LayerConfig},
    core::CapNet,
    layers::ConvLayer,
    capsule::{PrimaryCapsLayer, DigitCapsLayer},
};

/// CONSTRUCTEUR OPTIMISÉ
pub struct ModelBuilder {
    network_config: Option<NetworkConfig>,
    training_config: Option<TrainingConfig>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            network_config: None,
            training_config: None,
        }
    }

    pub fn with_network_config(mut self, config: NetworkConfig) -> Self {
        self.network_config = Some(config);
        self
    }

    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = Some(config);
        self
    }

    pub fn build(self) -> CapNet {
        let network_config = self.network_config.expect("Configuration réseau requise");
        let training_config = self.training_config.clone().unwrap_or_default();

        // VALIDATION OPTIMISÉE
        Self::validate_config(&network_config);

        let layers = Self::build_layers(&network_config);

        CapNet::new(network_config, training_config, layers)
    }

    fn validate_config(config: &NetworkConfig) {
        assert!(!config.layers.is_empty(), "Le réseau doit avoir au moins une couche");
        assert!(config.routing_iterations > 0, "Au moins une itération de routage requise");

        for (i, layer) in config.layers.iter().enumerate() {
            Self::validate_layer(i, layer);
        }
    }

    fn validate_layer(index: usize, layer: &LayerConfig) {
        match layer {
            LayerConfig::Conv2d { in_channels, out_channels, kernel_size, stride, .. } => {
                assert!(*in_channels > 0, "Couche {}: in_channels doit être > 0", index);
                assert!(*out_channels > 0, "Couche {}: out_channels doit être > 0", index);
                assert!(*kernel_size > 0, "Couche {}: kernel_size doit être > 0", index);
                assert!(*stride > 0, "Couche {}: stride doit être > 0", index);
            }
            LayerConfig::PrimaryCapsules { in_channels, capsule_config } => {
                assert!(*in_channels > 0, "Couche {}: in_channels doit être > 0", index);
                assert!(capsule_config.num_capsules > 0, "Couche {}: num_capsules doit être > 0", index);
                assert!(capsule_config.capsule_dim > 0, "Couche {}: capsule_dim doit être > 0", index);
            }
            LayerConfig::DigitCapsules { primary_capsules, primary_capsule_dim, capsule_config } => {
                assert!(*primary_capsules > 0, "Couche {}: primary_capsules doit être > 0", index);
                assert!(*primary_capsule_dim > 0, "Couche {}: primary_capsule_dim doit être > 0", index);
                assert!(capsule_config.num_capsules > 0, "Couche {}: num_capsules doit être > 0", index);
                assert!(capsule_config.capsule_dim > 0, "Couche {}: capsule_dim doit être > 0", index);
            }
        }
    }

    fn build_layers(config: &NetworkConfig) -> Vec<Layer> {
        let mut layers: Vec<Layer> = Vec::new();

        for layer_config in &config.layers {
            let layer = match layer_config {
                LayerConfig::Conv2d { in_channels, out_channels, kernel_size, stride, padding, .. } => {
                    Layer::Conv2d(ConvLayer::new(
                        *in_channels,
                        *out_channels,
                        *kernel_size,
                        *stride,
                        *padding,
                    ))
                }
                LayerConfig::PrimaryCapsules { in_channels, capsule_config } => {
                    Layer::PrimaryCapsules(PrimaryCapsLayer::new(
                        *in_channels,
                        capsule_config.num_capsules,
                        capsule_config.capsule_dim,
                        capsule_config.kernel_size,
                        capsule_config.stride,
                        capsule_config.padding,  // PADDING PASSÉ
                    ))
                }
                LayerConfig::DigitCapsules { primary_capsules, primary_capsule_dim, capsule_config } => {
                    Layer::DigitCapsules(DigitCapsLayer::new(
                        *primary_capsules,
                        *primary_capsule_dim,
                        capsule_config.num_capsules,
                        capsule_config.capsule_dim,
                        config.routing_iterations,
                    ))
                }
            };
            layers.push(layer);
        }

        layers
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// ENUMÉRATION OPTIMISÉE
pub enum Layer {
    Conv2d(ConvLayer),
    PrimaryCapsules(PrimaryCapsLayer),
    DigitCapsules(DigitCapsLayer),
}

impl Layer {
    pub fn forward(&self, input: &ndarray::ArrayView4<f32>) -> ndarray::Array4<f32> {
        match self {
            Layer::Conv2d(layer) => layer.forward(input),
            Layer::PrimaryCapsules(layer) => layer.forward(input),
            Layer::DigitCapsules(layer) => layer.forward(input),
        }
    }
}