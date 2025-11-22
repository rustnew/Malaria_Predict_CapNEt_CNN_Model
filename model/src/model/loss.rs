use ndarray::{Array4, ArrayView4, s};

/// Trait pour les fonctions de perte
pub trait LossFunction: Send + Sync {
    fn compute(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> f32;
    fn gradient(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> Array4<f32>;
}

/// Margin Loss pour CapsNet (Sabour et al., 2017)
pub struct MarginLoss {
    pub positive_margin: f32,   // m+ = 0.9
    pub negative_margin: f32,   // m- = 0.1
    pub down_weighting: f32,    // λ = 0.5
}

impl MarginLoss {
    pub fn new(positive_margin: f32, negative_margin: f32, down_weighting: f32) -> Self {
        Self {
            positive_margin,
            negative_margin,
            down_weighting,
        }
    }
    
    /// Calcule la norme d'une capsule (longueur du vecteur)
    fn capsule_norm(&self, capsule: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, num_capsules, spatial, dim) = capsule.dim();
        let mut norms = Array4::zeros((batch_size, num_capsules, spatial, 1));
        
        for b in 0..batch_size {
            for c in 0..num_capsules {
                for sp in 0..spatial {
                    let mut sum_sq = 0.0;
                    for d in 0..dim {
                        let val = capsule[[b, c, sp, d]];
                        sum_sq += val * val;
                    }
                    // Stabilité numérique
                    norms[[b, c, sp, 0]] = (sum_sq + 1e-8).sqrt();
                }
            }
        }
        
        norms
    }
}

impl LossFunction for MarginLoss {
    fn compute(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> f32 {
        let (batch_size, num_classes, _, _) = predictions.dim();
        let norms = self.capsule_norm(predictions);
        
        let mut total_loss = 0.0;
        
        for b in 0..batch_size {
            for c in 0..num_classes {
                let norm = norms[[b, c, 0, 0]];
                let target = targets[[b, c, 0, 0]];
                
                let loss = if target > 0.5 {
                    // Classe présente: L_k = T_k * max(0, m+ - ||v_k||)^2
                    (self.positive_margin - norm).max(0.0).powi(2)
                } else {
                    // Classe absente: L_k = λ * (1 - T_k) * max(0, ||v_k|| - m-)^2
                    self.down_weighting * (norm - self.negative_margin).max(0.0).powi(2)
                };
                
                total_loss += loss;
            }
        }
        
        // Moyenne sur le batch et les classes
        total_loss / (batch_size * num_classes) as f32
    }
    
    fn gradient(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, num_classes, spatial, dim) = predictions.dim();
        let norms = self.capsule_norm(predictions);
        
        let mut grad = Array4::zeros(predictions.dim());
        
        for b in 0..batch_size {
            for c in 0..num_classes {
                let norm = norms[[b, c, 0, 0]];
                let target = targets[[b, c, 0, 0]];
                
                // Éviter division par zéro
                if norm < 1e-7 {
                    continue;
                }
                
                let grad_factor = if target > 0.5 {
                    // Classe présente
                    if norm < self.positive_margin {
                        -2.0 * (self.positive_margin - norm) / norm
                    } else {
                        0.0
                    }
                } else {
                    // Classe absente
                    if norm > self.negative_margin {
                        2.0 * self.down_weighting * (norm - self.negative_margin) / norm
                    } else {
                        0.0
                    }
                };
                
                // Gradient pour chaque dimension de la capsule
                for d in 0..dim {
                    grad[[b, c, 0, d]] = grad_factor * predictions[[b, c, 0, d]];
                }
            }
        }
        
        // Normalisation
        grad / (batch_size * num_classes) as f32
    }
}

/// Perte de reconstruction (optionnelle)
pub struct ReconstructionLoss {
    pub weight: f32,  // Facteur de pondération (α)
}

impl ReconstructionLoss {
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }
}

impl LossFunction for ReconstructionLoss {
    fn compute(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> f32 {
        // MSE entre reconstruction et image originale
        let diff = predictions.to_owned() - targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        
        self.weight * mse
    }
    
    fn gradient(&self, predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> Array4<f32> {
        let diff = predictions.to_owned() - targets;
        let n = diff.len() as f32;
        
        diff * (2.0 * self.weight / n)
    }
}

/// Perte combinée: Margin Loss + Reconstruction Loss
pub struct CombinedLoss {
    margin_loss: MarginLoss,
    reconstruction_loss: Option<ReconstructionLoss>,
}

impl CombinedLoss {
    pub fn new(
        positive_margin: f32,
        negative_margin: f32,
        down_weighting: f32,
        reconstruction_weight: Option<f32>,
    ) -> Self {
        let margin_loss = MarginLoss::new(positive_margin, negative_margin, down_weighting);
        let reconstruction_loss = reconstruction_weight.map(|w| ReconstructionLoss::new(w));
        
        Self {
            margin_loss,
            reconstruction_loss,
        }
    }
    
    pub fn compute_total(
        &self,
        capsule_predictions: &ArrayView4<f32>,
        targets: &ArrayView4<f32>,
        reconstruction: Option<&ArrayView4<f32>>,
        original_images: Option<&ArrayView4<f32>>,
    ) -> f32 {
        let mut total_loss = self.margin_loss.compute(capsule_predictions, targets);
        
        if let (Some(recon), Some(original), Some(recon_loss)) = 
            (reconstruction, original_images, &self.reconstruction_loss) {
            total_loss += recon_loss.compute(&recon, &original);
        }
        
        total_loss
    }
    
    pub fn gradient_capsules(
        &self,
        predictions: &ArrayView4<f32>,
        targets: &ArrayView4<f32>,
    ) -> Array4<f32> {
        self.margin_loss.gradient(predictions, targets)
    }
}

/// Métriques d'évaluation
pub struct Metrics {
    pub accuracy: f32,
    pub precision: Vec<f32>,
    pub recall: Vec<f32>,
    pub f1_score: Vec<f32>,
    pub confusion_matrix: Vec<Vec<usize>>,
}

impl Metrics {
    pub fn compute(predictions: &ArrayView4<f32>, targets: &ArrayView4<f32>) -> Self {
        let (batch_size, num_classes, _, _) = predictions.dim();
        
        // Matrices de confusion
        let mut confusion = vec![vec![0; num_classes]; num_classes];
        let mut correct = 0;
        
        for b in 0..batch_size {
            // Trouver la classe prédite (capsule avec plus grande norme)
            let mut max_norm = 0.0;
            let mut pred_class = 0;
            
            for c in 0..num_classes {
                let mut norm_sq = 0.0;
                for d in 0..predictions.dim().3 {
                    let val = predictions[[b, c, 0, d]];
                    norm_sq += val * val;
                }
                let norm = norm_sq.sqrt();
                
                if norm > max_norm {
                    max_norm = norm;
                    pred_class = c;
                }
            }
            
            // Trouver la vraie classe
            let mut true_class = 0;
            for c in 0..num_classes {
                if targets[[b, c, 0, 0]] > 0.5 {
                    true_class = c;
                    break;
                }
            }
            
            confusion[true_class][pred_class] += 1;
            
            if pred_class == true_class {
                correct += 1;
            }
        }
        
        let accuracy = correct as f32 / batch_size as f32;
        
        // Calcul precision, recall, F1 par classe
        let mut precision = Vec::new();
        let mut recall = Vec::new();
        let mut f1_score = Vec::new();
        
        for c in 0..num_classes {
            let tp = confusion[c][c] as f32;
            let fp: f32 = (0..num_classes).filter(|&i| i != c)
                .map(|i| confusion[i][c] as f32).sum();
            let fn_: f32 = (0..num_classes).filter(|&i| i != c)
                .map(|i| confusion[c][i] as f32).sum();
            
            let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let rec = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
            
            precision.push(prec);
            recall.push(rec);
            f1_score.push(f1);
        }
        
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: confusion,
        }
    }
    
    pub fn print(&self) {
        println!("📊 MÉTRIQUES:");
        println!("   Accuracy: {:.4}", self.accuracy);
        
        for (i, ((p, r), f1)) in self.precision.iter()
            .zip(&self.recall)
            .zip(&self.f1_score)
            .enumerate() {
            println!("   Classe {}: Precision={:.4}, Recall={:.4}, F1={:.4}", 
                     i, p, r, f1);
        }
        
        println!("   Matrice de confusion:");
        for row in &self.confusion_matrix {
            println!("     {:?}", row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    
    #[test]
    fn test_margin_loss() {
        let loss_fn = MarginLoss::new(0.9, 0.1, 0.5);
        
        let predictions = Array4::from_elem((2, 2, 1, 8), 0.5);
        let mut targets = Array4::zeros((2, 2, 1, 1));
        targets[[0, 0, 0, 0]] = 1.0;
        targets[[1, 1, 0, 0]] = 1.0;
        
        let loss = loss_fn.compute(&predictions.view(), &targets.view());
        assert!(loss > 0.0);
    }
}