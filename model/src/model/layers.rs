use ndarray::{Array4, ArrayView4};
use rand::Rng;

/// Couche de convolution standard
pub struct ConvLayer {
    pub weights: Array4<f32>,
    pub biases: Array4<f32>,
    pub stride: usize,
    pub padding: usize,
    pub activation: Option<String>,
}

impl ConvLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut rng = rand::rng();  // CORRIGÉ
        
        // Initialisation He pour les poids
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();
        let weights_shape = (out_channels, in_channels, kernel_size, kernel_size);
        let weights = Array4::from_shape_fn(weights_shape, |_| rng.random::<f32>() * scale);  // CORRIGÉ
        
        let biases = Array4::zeros((out_channels, 1, 1, 1));
        
        Self {
            weights,
            biases,
            stride,
            padding,
            activation: Some("relu".to_string()),
        }
    }
    
    pub fn forward(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (out_channels, _, kernel_size, _) = self.weights.dim();
        
        let out_height = (in_height + 2 * self.padding - kernel_size) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_size) / self.stride + 1;
        
        println!("   🔍 ConvLayer:");
        println!("   - Entrée: {}x{}x{}", in_channels, in_height, in_width);
        println!("   - Sortie: {}x{}x{}", out_channels, out_height, out_width);
        println!("   - Kernel: {}, Stride: {}, Padding: {}", kernel_size, self.stride, self.padding);
        
        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
        
        // Implémentation basique de la convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;
                        
                        for ic in 0..in_channels {
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;
                                    
                                    if ih < in_height + self.padding && iw < in_width + self.padding {
                                        let input_val = if ih >= self.padding && iw >= self.padding &&
                                            ih < in_height + self.padding && iw < in_width + self.padding {
                                            input[[b, ic, ih - self.padding, iw - self.padding]]
                                        } else {
                                            0.0
                                        };
                                        
                                        sum += input_val * self.weights[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        
                        output[[b, oc, oh, ow]] = sum + self.biases[[oc, 0, 0, 0]];
                    }
                }
            }
        }
        
        // Application de la fonction d'activation
        if let Some(act) = &self.activation {
            match act.as_str() {
                "relu" => relu(&output),
                _ => output,
            }
        } else {
            output
        }
    }
}

/// Fonction d'activation ReLU
pub fn relu(x: &Array4<f32>) -> Array4<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Fonction squash pour les capsules
pub fn squash(vectors: &Array4<f32>) -> Array4<f32> {
    let mut result = vectors.clone();
    let (batch_size, num_capsules, spatial_points, capsule_dim) = vectors.dim();
    
    for b in 0..batch_size {
        for cap in 0..num_capsules {
            for sp in 0..spatial_points {
                let mut norm_squared = 0.0;
                
                // Calcul de la norme au carré
                for d in 0..capsule_dim {
                    let val = vectors[[b, cap, sp, d]];
                    norm_squared += val * val;
                }
                
                let norm = norm_squared.sqrt();
                let scale = norm_squared / (1.0 + norm_squared);
                
                // Application du squash
                for d in 0..capsule_dim {
                    let val = vectors[[b, cap, sp, d]];
                    result[[b, cap, sp, d]] = val * (scale / norm);
                }
            }
        }
    }
    
    result
}