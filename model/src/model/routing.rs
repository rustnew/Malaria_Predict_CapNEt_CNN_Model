use ndarray::{Array4, ArrayView4, ArrayViewMut4, s};  // Ajouter 's' ici

/// Implémentation du routage dynamique par accord entre les capsules
pub struct DynamicRouting {
    pub num_iterations: usize,
}

impl DynamicRouting {
    pub fn new(num_iterations: usize) -> Self {
        Self { num_iterations }
    }
    
    /// Algorithme de routage dynamique
    pub fn route(
        &self,
        predictions: &ArrayView4<f32>,  // [batch, digit_caps, primary_caps, dim]
    ) -> Array4<f32> {
        let (batch_size, digit_caps, primary_caps, _dim) = predictions.dim();  // Ajouter underscore
        
        // Initialisation des logits de couplage
        let mut coupling_logits = Array4::zeros((batch_size, digit_caps, primary_caps, 1));
        
        // Itérations de routage
        for _ in 0..self.num_iterations {
            // Application de softmax pour obtenir les coefficients de couplage
            let coupling_coefficients = self.softmax(&coupling_logits.view());
            
            // Calcul des capsules de sortie
            let outputs = self.calculate_outputs(predictions, &coupling_coefficients.view());
            
            // Mise à jour des accords
            self.update_agreement(predictions, &outputs.view(), coupling_logits.view_mut());
        }
        
        // Retourne les capsules de sortie finales
        let final_coefficients = self.softmax(&coupling_logits.view());
        self.calculate_outputs(predictions, &final_coefficients.view())
    }
    
    fn softmax(&self, logits: &ArrayView4<f32>) -> Array4<f32> {
        let mut result = Array4::zeros(logits.dim());
        
        // Softmax le long de l'axe des capsules de chiffres
        for b in 0..logits.dim().0 {
            let max_val = logits.slice(s![b, .., .., ..]).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = logits.slice(s![b, .., .., ..]).mapv(|x| (x - max_val).exp()).sum();
            
            for i in 0..logits.dim().1 {
                for j in 0..logits.dim().2 {
                    let val = (logits[[b, i, j, 0]] - max_val).exp() / exp_sum;
                    result[[b, i, j, 0]] = val;
                }
            }
        }
        
        result
    }
    
    fn calculate_outputs(
        &self,
        predictions: &ArrayView4<f32>,
        coupling_coefficients: &ArrayView4<f32>,
    ) -> Array4<f32> {
        let (batch_size, digit_caps, primary_caps, dim) = predictions.dim();
        let mut outputs = Array4::zeros((batch_size, digit_caps, 1, dim));
        
        for b in 0..batch_size {
            for dc in 0..digit_caps {
                for pc in 0..primary_caps {
                    let coefficient = coupling_coefficients[[b, dc, pc, 0]];
                    for d in 0..dim {
                        outputs[[b, dc, 0, d]] += coefficient * predictions[[b, dc, pc, d]];
                    }
                }
            }
        }
        
        outputs
    }
    
    fn update_agreement(
        &self,
        predictions: &ArrayView4<f32>,
        outputs: &ArrayView4<f32>,
        mut coupling_logits: ArrayViewMut4<f32>,
    ) {
        let (batch_size, digit_caps, primary_caps, dim) = predictions.dim();
        
        // Mise à jour des logits basée sur l'accord scalaire
        for b in 0..batch_size {
            for dc in 0..digit_caps {
                for pc in 0..primary_caps {
                    let mut agreement = 0.0;
                    for d in 0..dim {
                        agreement += predictions[[b, dc, pc, d]] * outputs[[b, dc, 0, d]];
                    }
                    coupling_logits[[b, dc, pc, 0]] += agreement;
                }
            }
        }
    }
}