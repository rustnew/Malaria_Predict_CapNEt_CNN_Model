use ndarray::{Array4, ArrayView4, Axis};
use rayon::prelude::*;

/// Routage dynamique optimisé avec agreement
pub struct DynamicRouting {
    pub num_iterations: usize,
}

impl DynamicRouting {
    pub fn new(num_iterations: usize) -> Self {
        Self { num_iterations }
    }
    
    /// Routage avec retour des coefficients de couplage
    pub fn route_with_coeffs(
        &self,
        predictions: &ArrayView4<f32>,
    ) -> (Array4<f32>, Array4<f32>) {
        let (batch_size, output_caps, input_caps, _dim) = predictions.dim();
        
        // Initialisation des logits (b_ij)
        let mut logits = Array4::zeros((batch_size, output_caps, input_caps, 1));
        let mut coupling_coeffs = Array4::zeros((batch_size, output_caps, input_caps, 1));
        
        for iteration in 0..self.num_iterations {
            // Softmax pour obtenir les coefficients de couplage (c_ij)
            coupling_coeffs = self.softmax_parallel(&logits.view());
            
            // Calcul des capsules de sortie (s_j = Σ c_ij * û_j|i)
            let outputs = self.weighted_sum_parallel(predictions, &coupling_coeffs.view());
            
            // Squash sur les capsules de sortie (v_j = squash(s_j))
            let squashed_outputs = self.squash_output(&outputs.view());
            
            // Mise à jour des logits si ce n'est pas la dernière itération
            if iteration < self.num_iterations - 1 {
                self.update_logits_parallel(
                    predictions,
                    &squashed_outputs.view(),
                    &mut logits,
                );
            }
        }
        
        // Calcul final des sorties
        let final_outputs = self.weighted_sum_parallel(predictions, &coupling_coeffs.view());
        let final_squashed = self.squash_output(&final_outputs.view());
        
        (final_squashed, coupling_coeffs)
    }
    
    /// Softmax parallélisé sur l'axe des capsules de sortie
    fn softmax_parallel(&self, logits: &ArrayView4<f32>) -> Array4<f32> {
        let (_batch_size, output_caps, input_caps, _) = logits.dim();
        let mut result = Array4::zeros(logits.dim());
        
        // Parallélisation par batch
        result.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut result_batch)| {
                // Pour chaque capsule d'entrée
                for ic in 0..input_caps {
                    // Trouver le max pour la stabilité numérique
                    let mut max_val = f32::NEG_INFINITY;
                    for oc in 0..output_caps {
                        max_val = max_val.max(logits[[b, oc, ic, 0]]);
                    }
                    
                    // Calculer exp et somme
                    let mut exp_sum = 0.0;
                    let mut exp_vals = vec![0.0; output_caps];
                    
                    for oc in 0..output_caps {
                        let exp_val = (logits[[b, oc, ic, 0]] - max_val).exp();
                        exp_vals[oc] = exp_val;
                        exp_sum += exp_val;
                    }
                    
                    // Normaliser
                    for oc in 0..output_caps {
                        result_batch[[oc, ic, 0]] = exp_vals[oc] / (exp_sum + 1e-8);
                    }
                }
            });
        
        result
    }
    
    /// Somme pondérée parallélisée
    fn weighted_sum_parallel(
        &self,
        predictions: &ArrayView4<f32>,
        coupling_coeffs: &ArrayView4<f32>,
    ) -> Array4<f32> {
        let (batch_size, output_caps, input_caps, dim) = predictions.dim();
        let mut outputs = Array4::zeros((batch_size, output_caps, 1, dim));
        
        // Parallélisation par batch et capsule de sortie
        outputs.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut output_batch)| {
                for oc in 0..output_caps {
                    for d in 0..dim {
                        let mut sum = 0.0;
                        
                        for ic in 0..input_caps {
                            let coeff = coupling_coeffs[[b, oc, ic, 0]];
                            let pred = predictions[[b, oc, ic, d]];
                            sum += coeff * pred;
                        }
                        
                        output_batch[[oc, 0, d]] = sum;
                    }
                }
            });
        
        outputs
    }
    
    /// Squash sur les capsules de sortie
    fn squash_output(&self, outputs: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, num_caps, _, dim) = outputs.dim();
        let mut result = outputs.to_owned();
        
        for b in 0..batch_size {
            for c in 0..num_caps {
                let mut norm_squared = 0.0;
                
                for d in 0..dim {
                    let val = outputs[[b, c, 0, d]];
                    norm_squared += val * val;
                }
                
                // Stabilité numérique
                let norm = (norm_squared + 1e-8).sqrt();
                let scale = norm_squared / (1.0 + norm_squared);
                let factor = scale / norm;
                
                for d in 0..dim {
                    result[[b, c, 0, d]] = outputs[[b, c, 0, d]] * factor;
                }
            }
        }
        
        result
    }
    
    /// Mise à jour parallélisée des logits par agreement
    fn update_logits_parallel(
        &self,
        predictions: &ArrayView4<f32>,
        outputs: &ArrayView4<f32>,
        logits: &mut Array4<f32>,
    ) {
        let (batch_size, output_caps, input_caps, dim) = predictions.dim();
        
        // Parallélisation par batch
        for b in 0..batch_size {
            for oc in 0..output_caps {
                for ic in 0..input_caps {
                    // Agreement: produit scalaire entre prédiction et sortie
                    let mut agreement = 0.0;
                    
                    for d in 0..dim {
                        agreement += predictions[[b, oc, ic, d]] * outputs[[b, oc, 0, d]];
                    }
                    
                    logits[[b, oc, ic, 0]] += agreement;
                }
            }
        }
    }
    
    /// Version simple sans coefficients (pour compatibilité)
    pub fn route(&self, predictions: &ArrayView4<f32>) -> Array4<f32> {
        let (outputs, _) = self.route_with_coeffs(predictions);
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    
    #[test]
    fn test_routing_dimensions() {
        let routing = DynamicRouting::new(3);
        let predictions = Array4::ones((2, 10, 1152, 16));
        
        let output = routing.route(&predictions.view());
        
        assert_eq!(output.dim(), (2, 10, 1, 16));
    }
    
    #[test]
    fn test_softmax_sums_to_one() {
        let routing = DynamicRouting::new(3);
        let logits = Array4::ones((2, 10, 32, 1));
        
        let coeffs = routing.softmax_parallel(&logits.view());
        
        // Vérifier que la somme sur les capsules de sortie = 1
        for b in 0..2 {
            for ic in 0..32 {
                let sum: f32 = (0..10).map(|oc| coeffs[[b, oc, ic, 0]]).sum();
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }
    }
}