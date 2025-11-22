use ndarray::{Array4, ArrayView4, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;
use crate::model::layers::{Layer, squash, squash_gradient, ConvLayer};
use crate::model::routing::DynamicRouting;
use crate::model::config::Activation;

/// Couche de capsules primaires avec backpropagation
pub struct PrimaryCapsLayer {
    conv_layers: Vec<ConvLayer>,
    num_capsules: usize,
    capsule_dim: usize,
    
    // Cache pour backprop
    input_cache: Option<Array4<f32>>,
    pre_squash_cache: Option<Array4<f32>>,
}

impl PrimaryCapsLayer {
    pub fn new(
        in_channels: usize,
        num_capsules: usize,
        capsule_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let total_convs = num_capsules * capsule_dim;
        let mut conv_layers = Vec::with_capacity(total_convs);
        
        for _ in 0..total_convs {
            conv_layers.push(ConvLayer::new(
                in_channels,
                1,
                kernel_size,
                stride,
                padding,
                Activation::None, // Pas d'activation avant squash
            ));
        }
        
        Self {
            conv_layers,
            num_capsules,
            capsule_dim,
            input_cache: None,
            pre_squash_cache: None,
        }
    }
}

impl Layer for PrimaryCapsLayer {
    fn forward(&mut self, input: &ArrayView4<f32>) -> Array4<f32> {
        self.input_cache = Some(input.to_owned());
        
        let (batch_size, _, _, _) = input.dim();
        
        // Obtenir les dimensions de sortie de la première conv
        let test_output = self.conv_layers[0].forward(input);
        let (_, _, out_height, out_width) = test_output.dim();
        let spatial_points = out_height * out_width;
        
        // Préparer le tenseur de capsules
        let mut capsules = Array4::zeros((
            batch_size,
            self.num_capsules,
            spatial_points,
            self.capsule_dim,
        ));
        
        // Appliquer toutes les convolutions en parallèle
        let conv_outputs: Vec<_> = self.conv_layers.par_iter_mut()
            .map(|conv| conv.forward(input))
            .collect();
        
        // Réorganiser en capsules
        for cap_idx in 0..self.num_capsules {
            for dim_idx in 0..self.capsule_dim {
                let conv_idx = cap_idx * self.capsule_dim + dim_idx;
                let conv_output = &conv_outputs[conv_idx];
                
                for b in 0..batch_size {
                    for h in 0..out_height {
                        for w in 0..out_width {
                            let spatial_idx = h * out_width + w;
                            capsules[[b, cap_idx, spatial_idx, dim_idx]] = 
                                conv_output[[b, 0, h, w]];
                        }
                    }
                }
            }
        }
        
        self.pre_squash_cache = Some(capsules.clone());
        squash(&capsules)
    }

    fn backward(&mut self, grad_output: &ArrayView4<f32>) -> Array4<f32> {
        let input = self.input_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        let pre_squash = self.pre_squash_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        
        // Gradient à travers squash
        let grad_pre_squash = squash_gradient(&pre_squash.view(), &grad_output.to_owned().view());
        
        let (batch_size, _, in_height, in_width) = input.dim();
        let (_, _, spatial_points, _) = grad_pre_squash.dim();
        
        // Calculer dimensions de sortie conv
        let out_height = (spatial_points as f32).sqrt() as usize;
        let out_width = spatial_points / out_height;
        
        // Réorganiser le gradient pour chaque conv
        let mut grad_input = Array4::zeros((batch_size, input.dim().1, in_height, in_width));
        
        for cap_idx in 0..self.num_capsules {
            for dim_idx in 0..self.capsule_dim {
                let conv_idx = cap_idx * self.capsule_dim + dim_idx;
                
                // Préparer le gradient pour cette convolution
                let mut grad_conv = Array4::zeros((batch_size, 1, out_height, out_width));
                
                for b in 0..batch_size {
                    for h in 0..out_height {
                        for w in 0..out_width {
                            let spatial_idx = h * out_width + w;
                            grad_conv[[b, 0, h, w]] = 
                                grad_pre_squash[[b, cap_idx, spatial_idx, dim_idx]];
                        }
                    }
                }
                
                // Backprop à travers la convolution
                let grad_from_conv = self.conv_layers[conv_idx].backward(&grad_conv.view());
                grad_input += &grad_from_conv;
            }
        }
        
        grad_input
    }

    fn update_weights(&mut self, learning_rate: f32) {
        self.conv_layers.par_iter_mut()
            .for_each(|conv| conv.update_weights(learning_rate));
    }

    fn zero_grad(&mut self) {
        self.conv_layers.iter_mut()
            .for_each(|conv| conv.zero_grad());
    }
}

/// Couche de capsules de sortie avec routage dynamique
pub struct DigitCapsLayer {
    weights: Array4<f32>,
    routing: DynamicRouting,
    num_output_capsules: usize,
    output_capsule_dim: usize,
    
    // Cache pour backprop
    input_cache: Option<Array4<f32>>,
    predictions_cache: Option<Array4<f32>>,
    coupling_coeffs_cache: Option<Array4<f32>>,
    weight_grad: Array4<f32>,
}

impl DigitCapsLayer {
    pub fn new(
        input_capsules: usize,
        input_capsule_dim: usize,
        output_capsules: usize,
        output_capsule_dim: usize,
        routing_iterations: usize,
    ) -> Self {
        // Initialisation Xavier - CORRIGÉ AVEC .expect()
        let scale = (2.0 / (input_capsule_dim + output_capsule_dim) as f32).sqrt();
        let weights = Array4::random(
            (output_capsules, input_capsules, output_capsule_dim, input_capsule_dim),
            Uniform::new(-scale, scale).expect("Invalid uniform distribution range")
        );
        
        let weight_grad = Array4::zeros(weights.dim());
        
        Self {
            weights,
            routing: DynamicRouting::new(routing_iterations),
            num_output_capsules: output_capsules,
            output_capsule_dim: output_capsule_dim,
            input_cache: None,
            predictions_cache: None,
            coupling_coeffs_cache: None,
            weight_grad,
        }
    }
    
    fn compute_predictions(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, input_caps, spatial_points, input_dim) = input.dim();
        let total_input_caps = input_caps * spatial_points;
        
        let mut predictions = Array4::zeros((
            batch_size,
            self.num_output_capsules,
            total_input_caps,
            self.output_capsule_dim,
        ));
        
        // Calcul parallélisé des prédictions
        predictions.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut pred_batch)| {
                for output_cap in 0..self.num_output_capsules {
                    for input_cap in 0..input_caps {
                        for sp in 0..spatial_points {
                            let input_idx = input_cap * spatial_points + sp;
                            
                            // Multiplication matricielle: W * input_capsule
                            for out_dim in 0..self.output_capsule_dim {
                                let mut sum = 0.0;
                                for in_dim in 0..input_dim {
                                    sum += self.weights[[output_cap, input_cap, out_dim, in_dim]]
                                        * input[[b, input_cap, sp, in_dim]];
                                }
                                pred_batch[[output_cap, input_idx, out_dim]] = sum;
                            }
                        }
                    }
                }
            });
        
        predictions
    }
}

impl Layer for DigitCapsLayer {
    fn forward(&mut self, input: &ArrayView4<f32>) -> Array4<f32> {
        self.input_cache = Some(input.to_owned());
        
        // Calculer les prédictions
        let predictions = self.compute_predictions(input);
        self.predictions_cache = Some(predictions.clone());
        
        // Routage dynamique
        let (output, coupling_coeffs) = self.routing.route_with_coeffs(&predictions.view());
        self.coupling_coeffs_cache = Some(coupling_coeffs);
        
        output
    }

    fn backward(&mut self, grad_output: &ArrayView4<f32>) -> Array4<f32> {
        let input = self.input_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        let _predictions = self.predictions_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        let coupling_coeffs = self.coupling_coeffs_cache.as_ref()
            .expect("Forward doit être appelé avant backward");
        
        let (batch_size, input_caps, spatial_points, input_dim) = input.dim();
        let _total_input_caps = input_caps * spatial_points;
        
        // Gradient des poids
        self.weight_grad.fill(0.0);
        
        for b in 0..batch_size {
            for output_cap in 0..self.num_output_capsules {
                for input_cap in 0..input_caps {
                    for sp in 0..spatial_points {
                        let input_idx = input_cap * spatial_points + sp;
                        let coeff = coupling_coeffs[[b, output_cap, input_idx, 0]];
                        
                        for out_dim in 0..self.output_capsule_dim {
                            let grad_val = grad_output[[b, output_cap, 0, out_dim]];
                            
                            for in_dim in 0..input_dim {
                                self.weight_grad[[output_cap, input_cap, out_dim, in_dim]] +=
                                    coeff * grad_val * input[[b, input_cap, sp, in_dim]];
                            }
                        }
                    }
                }
            }
        }
        
        self.weight_grad /= batch_size as f32;
        
        // Gradient par rapport à l'entrée
        let mut grad_input = Array4::zeros(input.dim());
        
        for b in 0..batch_size {
            for input_cap in 0..input_caps {
                for sp in 0..spatial_points {
                    let input_idx = input_cap * spatial_points + sp;
                    
                    for in_dim in 0..input_dim {
                        let mut sum = 0.0;
                        
                        for output_cap in 0..self.num_output_capsules {
                            let coeff = coupling_coeffs[[b, output_cap, input_idx, 0]];
                            
                            for out_dim in 0..self.output_capsule_dim {
                                let grad_val = grad_output[[b, output_cap, 0, out_dim]];
                                sum += coeff * grad_val 
                                    * self.weights[[output_cap, input_cap, out_dim, in_dim]];
                            }
                        }
                        
                        grad_input[[b, input_cap, sp, in_dim]] = sum;
                    }
                }
            }
        }
        
        grad_input
    }

    fn update_weights(&mut self, learning_rate: f32) {
        self.weights.scaled_add(-learning_rate, &self.weight_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grad.fill(0.0);
    }
}