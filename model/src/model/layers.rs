use ndarray::{Array4, ArrayView4, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;
use crate::model::config::Activation;

/// Trait pour toutes les couches avec forward et backward
pub trait Layer: Send + Sync {
    fn forward(&mut self, input: &ArrayView4<f32>) -> Array4<f32>;
    fn backward(&mut self, grad_output: &ArrayView4<f32>) -> Array4<f32>;
    fn update_weights(&mut self, learning_rate: f32);
    fn zero_grad(&mut self);
}

/// Couche de convolution optimisée avec backpropagation
pub struct ConvLayer {
    pub weights: Array4<f32>,
    pub biases: Array4<f32>,
    pub stride: usize,
    pub padding: usize,
    pub activation: Activation,
    
    // Pour backprop
    input_cache: Option<Array4<f32>>,
    weight_grad: Array4<f32>,
    bias_grad: Array4<f32>,
    pre_activation_cache: Option<Array4<f32>>,
}

impl ConvLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
    ) -> Self {
        // Initialisation He
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();
        let weights = Array4::random(
            (out_channels, in_channels, kernel_size, kernel_size),
            Uniform::new(-scale, scale).expect("Invalid uniform distribution range")
        );
        
        let biases = Array4::zeros((out_channels, 1, 1, 1));
        let weight_grad = Array4::zeros(weights.dim());
        let bias_grad = Array4::zeros(biases.dim());
        
        Self {
            weights,
            biases,
            stride,
            padding,
            activation,
            input_cache: None,
            weight_grad,
            bias_grad,
            pre_activation_cache: None,
        }
    }

    /// Convolution optimisée avec im2col
    fn convolve_im2col(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (out_channels, _, kernel_size, _) = self.weights.dim();
        
        let out_height = (in_height + 2 * self.padding - kernel_size) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_size) / self.stride + 1;
        
        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
        
        // Padding de l'entrée si nécessaire
        let padded = if self.padding > 0 {
            self.pad_input(input)
        } else {
            input.to_owned()
        };
        
        // Convolution parallélisée par batch
        output.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut out_batch)| {
                let input_batch = padded.index_axis(Axis(0), b);
                
                for oc in 0..out_channels {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let mut sum = 0.0;
                            
                            let ih_start = oh * self.stride;
                            let iw_start = ow * self.stride;
                            
                            for ic in 0..in_channels {
                                for kh in 0..kernel_size {
                                    for kw in 0..kernel_size {
                                        let ih = ih_start + kh;
                                        let iw = iw_start + kw;
                                        
                                        sum += input_batch[[ic, ih, iw]] 
                                            * self.weights[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                            
                            out_batch[[oc, oh, ow]] = sum + self.biases[[oc, 0, 0, 0]];
                        }
                    }
                }
            });
        
        output
    }

    fn pad_input(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, channels, height, width) = input.dim();
        let p = self.padding;
        
        let mut padded = Array4::zeros((
            batch_size, 
            channels, 
            height + 2 * p, 
            width + 2 * p
        ));
        
        for b in 0..batch_size {
            for c in 0..channels {
                padded.slice_mut(s![b, c, p..height+p, p..width+p])
                    .assign(&input.slice(s![b, c, .., ..]));
            }
        }
        
        padded
    }

    fn apply_activation(&self, x: &Array4<f32>) -> Array4<f32> {
        match self.activation {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::None => x.clone(),
        }
    }

    fn activation_derivative(&self, x: &Array4<f32>) -> Array4<f32> {
        match self.activation {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { alpha }),
            Activation::Sigmoid => {
                let sig = self.apply_activation(x);
                &sig * &sig.mapv(|v| 1.0 - v)
            }
            Activation::Tanh => {
                let tanh = x.mapv(|v| v.tanh());
                tanh.mapv(|v| 1.0 - v * v)
            }
            Activation::None => Array4::ones(x.dim()),
        }
    }
}

impl Layer for ConvLayer {
    fn forward(&mut self, input: &ArrayView4<f32>) -> Array4<f32> {
        // Cache pour backward
        self.input_cache = Some(input.to_owned());
        
        // Convolution
        let pre_activation = self.convolve_im2col(input);
        self.pre_activation_cache = Some(pre_activation.clone());
        
        // Activation
        self.apply_activation(&pre_activation)
    }

    fn backward(&mut self, grad_output: &ArrayView4<f32>) -> Array4<f32> {
        let input = self.input_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        let pre_activation = self.pre_activation_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        
        // Gradient à travers l'activation
        let activation_grad = self.activation_derivative(pre_activation);
        let grad = grad_output.to_owned() * activation_grad;
        
        let (batch_size, out_channels, out_height, out_width) = grad.dim();
        let (_, in_channels, kernel_size, _) = self.weights.dim();
        
        // Gradient des poids et biais
        self.weight_grad.fill(0.0);
        self.bias_grad.fill(0.0);
        
        let padded_input = if self.padding > 0 {
            self.pad_input(&input.view())
        } else {
            input.clone()
        };
        
        // Calcul du gradient des poids
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_val = grad[[b, oc, oh, ow]];
                        
                        let ih_start = oh * self.stride;
                        let iw_start = ow * self.stride;
                        
                        for ic in 0..in_channels {
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    self.weight_grad[[oc, ic, kh, kw]] += 
                                        grad_val * padded_input[[b, ic, ih_start + kh, iw_start + kw]];
                                }
                            }
                        }
                        
                        self.bias_grad[[oc, 0, 0, 0]] += grad_val;
                    }
                }
            }
        }
        
        // Normalisation par batch size
        self.weight_grad /= batch_size as f32;
        self.bias_grad /= batch_size as f32;
        
        // Gradient par rapport à l'entrée (pour la couche précédente)
        let (_, _, in_height, in_width) = input.dim();
        let mut grad_input = Array4::zeros((batch_size, in_channels, in_height, in_width));
        
        for b in 0..batch_size {
            for ic in 0..in_channels {
                for ih in 0..in_height {
                    for iw in 0..in_width {
                        let mut sum = 0.0;
                        
                        for oc in 0..out_channels {
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    let oh = (ih + self.padding).saturating_sub(kh) / self.stride;
                                    let ow = (iw + self.padding).saturating_sub(kw) / self.stride;
                                    
                                    if oh < out_height && ow < out_width {
                                        sum += grad[[b, oc, oh, ow]] * self.weights[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        
                        grad_input[[b, ic, ih, iw]] = sum;
                    }
                }
            }
        }
        
        grad_input
    }

    fn update_weights(&mut self, learning_rate: f32) {
        self.weights.scaled_add(-learning_rate, &self.weight_grad);
        self.biases.scaled_add(-learning_rate, &self.bias_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grad.fill(0.0);
        self.bias_grad.fill(0.0);
    }
}

/// Fonction squash pour capsules avec gradient
pub fn squash(vectors: &Array4<f32>) -> Array4<f32> {
    let (batch_size, num_capsules, spatial_points, capsule_dim) = vectors.dim();
    let mut result = vectors.clone();
    
    for b in 0..batch_size {
        for cap in 0..num_capsules {
            for sp in 0..spatial_points {
                let mut norm_squared = 0.0;
                
                for d in 0..capsule_dim {
                    let val = vectors[[b, cap, sp, d]];
                    norm_squared += val * val;
                }
                
                // Stabilité numérique
                let norm = (norm_squared + 1e-8).sqrt();
                let scale = norm_squared / (1.0 + norm_squared);
                let factor = scale / norm;
                
                for d in 0..capsule_dim {
                    result[[b, cap, sp, d]] = vectors[[b, cap, sp, d]] * factor;
                }
            }
        }
    }
    
    result
}

pub fn squash_gradient(vectors: &ArrayView4<f32>, grad_output: &ArrayView4<f32>) -> Array4<f32> {
    let (batch_size, num_capsules, spatial_points, capsule_dim) = vectors.dim();
    let mut grad_input = Array4::zeros(vectors.dim());
    
    for b in 0..batch_size {
        for cap in 0..num_capsules {
            for sp in 0..spatial_points {
                let mut norm_squared = 0.0;
                let mut vector = Vec::with_capacity(capsule_dim);
                
                for d in 0..capsule_dim {
                    let val = vectors[[b, cap, sp, d]];
                    vector.push(val);
                    norm_squared += val * val;
                }
                
                let norm = (norm_squared + 1e-8).sqrt();
                let scale = norm_squared / (1.0 + norm_squared);
                
                // Jacobien du squash
                for d1 in 0..capsule_dim {
                    let mut grad_sum = 0.0;
                    
                    for d2 in 0..capsule_dim {
                        let grad_out = grad_output[[b, cap, sp, d2]];
                        
                        let jacobian = if d1 == d2 {
                            scale / norm + 2.0 * vector[d1] * vector[d2] * (1.0 / (1.0 + norm_squared).powi(2) - scale / (norm * norm_squared))
                        } else {
                            2.0 * vector[d1] * vector[d2] * (1.0 / (1.0 + norm_squared).powi(2) - scale / (norm * norm_squared))
                        };
                        
                        grad_sum += grad_out * jacobian;
                    }
                    
                    grad_input[[b, cap, sp, d1]] = grad_sum;
                }
            }
        }
    }
    
    grad_input
}