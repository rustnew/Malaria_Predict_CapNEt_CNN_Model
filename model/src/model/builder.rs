use super::{
    config::{NetworkConfig, TrainingConfig, LayerConfig},
    core::CapNet,
    layers::{ConvLayer, Layer},
    capsule::{PrimaryCapsLayer, DigitCapsLayer},
};

/// Constructeur de modèle
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

    pub fn build(self) -> Result<CapNet, String> {
        let network_config = self.network_config
            .ok_or_else(|| "Configuration réseau requise".to_string())?;
        let training_config = self.training_config
            .unwrap_or_default();

        // Validation
        network_config.validate()?;
        training_config.validate()?;

        let layers = Self::build_layers(&network_config)?;

        Ok(CapNet::new(network_config, training_config, layers))
    }

    fn build_layers(config: &NetworkConfig) -> Result<Vec<Box<dyn Layer>>, String> {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        for (i, layer_config) in config.layers.iter().enumerate() {
            let layer: Box<dyn Layer> = match layer_config {
                LayerConfig::Conv2d { 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride, 
                    padding,
                    activation,
                } => {
                    Box::new(ConvLayer::new(
                        *in_channels,
                        *out_channels,
                        *kernel_size,
                        *stride,
                        *padding,
                        *activation,
                    ))
                }
                
                LayerConfig::PrimaryCapsules { in_channels, capsule_config } => {
                    Box::new(PrimaryCapsLayer::new(
                        *in_channels,
                        capsule_config.num_capsules,
                        capsule_config.capsule_dim,
                        capsule_config.kernel_size,
                        capsule_config.stride,
                        capsule_config.padding,
                    ))
                }
                
                LayerConfig::DigitCapsules { 
                    input_capsules,
                    input_capsule_dim,
                    output_capsules,
                    output_capsule_dim,
                } => {
                    Box::new(DigitCapsLayer::new(
                        *input_capsules,
                        *input_capsule_dim,
                        *output_capsules,
                        *output_capsule_dim,
                        config.routing_iterations,
                    ))
                }
            };
            
            layers.push(layer);
        }

        if layers.is_empty() {
            return Err("Le réseau doit avoir au moins une couche".to_string());
        }

        Ok(layers)
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}