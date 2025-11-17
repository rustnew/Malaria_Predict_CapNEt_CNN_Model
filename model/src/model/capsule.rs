use ndarray::{Array4, ArrayView4};
use rand::Rng;
use super::layers::{squash, ConvLayer};
use super::routing::DynamicRouting;

/// Couche de capsules primaires OPTIMISÉE
pub struct PrimaryCapsLayer {
    pub conv_layers: Vec<ConvLayer>,
    pub num_capsules: usize,
    pub capsule_dim: usize,
}

impl PrimaryCapsLayer {
    pub fn new(
        in_channels: usize,
        num_capsules: usize,
        capsule_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,  // NOUVEAU PARAMÈTRE
    ) -> Self {
        let conv_layers = (0..num_capsules * capsule_dim)
            .map(|_| ConvLayer::new(in_channels, 1, kernel_size, stride, padding))
            .collect();
        
        Self {
            conv_layers,
            num_capsules,
            capsule_dim,
        }
    }
    
    pub fn forward(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, _, in_height, in_width) = input.dim();
        
        // TEST: Obtenir les dimensions RÉELLES
        let test_output = self.conv_layers[0].forward(input);
        let (_, _, actual_height, actual_width) = test_output.dim();
        
        println!("   🔍 Capsules Primaires:");
        println!("   - Entrée: {}x{}", in_height, in_width);
        println!("   - Sortie conv: {}x{}", actual_height, actual_width);
        println!("   - Spatial points: {}", actual_height * actual_width);
        
        let mut capsules = Array4::zeros((
            batch_size,
            self.num_capsules,
            actual_height * actual_width,
            self.capsule_dim,
        ));
        
        // Application OPTIMISÉE des convolutions
        for cap_idx in 0..self.num_capsules {
            for dim_idx in 0..self.capsule_dim {
                let conv_idx = cap_idx * self.capsule_dim + dim_idx;
                let conv_output = self.conv_layers[conv_idx].forward(input);
                
                // Réorganisation OPTIMISÉE
                for b in 0..batch_size {
                    for h in 0..actual_height {
                        for w in 0..actual_width {
                            let spatial_idx = h * actual_width + w;
                            capsules[[b, cap_idx, spatial_idx, dim_idx]] = 
                                conv_output[[b, 0, h, w]];
                        }
                    }
                }
            }
        }
        
        println!("   ✅ Capsules shape: {:?}", capsules.dim());
        squash(&capsules)
    }
}

/// Couche de capsules de chiffres OPTIMISÉE
pub struct DigitCapsLayer {
    pub routing: DynamicRouting,
    pub num_capsules: usize,
    pub capsule_dim: usize,
    pub weights: Array4<f32>,
}

impl DigitCapsLayer {
    
    pub fn new(
        primary_capsules: usize,
        primary_capsule_dim: usize,
        digit_capsules: usize,
        digit_capsule_dim: usize,
        routing_iterations: usize,
    ) -> Self {
        let weights_shape = (digit_capsules, primary_capsules, digit_capsule_dim, primary_capsule_dim);
        let mut weights = Array4::zeros(weights_shape);
        
        // Initialisation OPTIMISÉE
        let mut rng = rand::rng();
        for i in 0..digit_capsules {
            for j in 0..primary_capsules {
                for k in 0..digit_capsule_dim {
                    for l in 0..primary_capsule_dim {
                        weights[[i, j, k, l]] = rng.random::<f32>() * 0.01;
                    }
                }
            }
        }
        
        Self {
            routing: DynamicRouting::new(routing_iterations),
            num_capsules: digit_capsules,
            capsule_dim: digit_capsule_dim,
            weights,
        }
    }
    
    pub fn forward(&self, primary_capsules: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, primary_caps, spatial_points, primary_dim) = primary_capsules.dim();
        
        println!("   🔍 Capsules Chiffres:");
        println!("   - Entrée: {} capsules, {} points, dim {}", primary_caps, spatial_points, primary_dim);
        
        let predictions_shape = (batch_size, self.num_capsules, primary_caps * spatial_points, self.capsule_dim);
        let mut predictions = Array4::zeros(predictions_shape);
        
        // Transformation OPTIMISÉE
        for b in 0..batch_size {
            for dc in 0..self.num_capsules {
                for pc in 0..primary_caps {
                    for sp in 0..spatial_points {
                        let mut transformed = vec![0.0; self.capsule_dim];
                        
                        for i in 0..self.capsule_dim {
                            for j in 0..primary_dim {
                                transformed[i] += self.weights[[dc, pc, i, j]] * 
                                    primary_capsules[[b, pc, sp, j]];
                            }
                        }
                        
                        for d in 0..self.capsule_dim {
                            let pred_idx = pc * spatial_points + sp;
                            predictions[[b, dc, pred_idx, d]] = transformed[d];
                        }
                    }
                }
            }
        }
        
        let output = self.routing.route(&predictions.view());
        println!("   ✅ Sortie capsules: {:?}", output.dim());
        output
    }
}