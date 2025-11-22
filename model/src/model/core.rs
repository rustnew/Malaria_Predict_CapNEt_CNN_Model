use super::{
    config::{NetworkConfig, TrainingConfig},
    layers::Layer,
    optimizer::{Optimizer, Adam, SGD},
    loss::{CombinedLoss, Metrics},
};
use ndarray::{Array4, ArrayView4, s};

/// Modèle CapsNet principal
pub struct CapNet {
    pub network_config: NetworkConfig,
    pub training_config: TrainingConfig,
    pub layers: Vec<Box<dyn Layer>>,
    pub history: TrainingHistory,
    pub state: ModelState,
    pub loss_fn: CombinedLoss,
}

#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub train_accuracy: Vec<f32>,
    pub val_accuracy: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ModelState {
    pub is_trained: bool,
    pub best_loss: f32,
    pub current_epoch: usize,
    pub early_stopping_counter: usize,
}

impl CapNet {
    pub fn new(
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        layers: Vec<Box<dyn Layer>>,
    ) -> Self {
        // Créer la fonction de perte
        let loss_fn = CombinedLoss::new(
            training_config.loss_config.positive_margin,
            training_config.loss_config.negative_margin,
            training_config.loss_config.down_weighting,
            Some(training_config.loss_config.reconstruction_weight),
        );

        Self {
            network_config,
            training_config,
            layers,
            history: TrainingHistory::new(),
            state: ModelState::new(),
            loss_fn,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &ArrayView4<f32>) -> Array4<f32> {
        let mut output = input.to_owned();
        
        for layer in &mut self.layers {
            output = layer.forward(&output.view());
        }
        
        output
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &ArrayView4<f32>) -> Array4<f32> {
        let mut grad = grad_output.to_owned();
        
        // Backprop à travers toutes les couches en ordre inverse
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad.view());
        }
        
        grad
    }

    /// Mise à jour des poids
    pub fn update_weights(&mut self, optimizer: &mut dyn Optimizer) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let param_id = format!("layer_{}", i);
            layer.update_weights(optimizer.get_lr());
        }
    }

    /// Entraînement complet
    pub fn train(
        mut self,
        train_data: Array4<f32>,
        train_labels: Array4<f32>,
        val_data: Array4<f32>,
        val_labels: Array4<f32>,
    ) -> Self {
        println!("🎯 DÉBUT DE L'ENTRAÎNEMENT");
        println!("   Échantillons train: {}", train_data.dim().0);
        println!("   Échantillons validation: {}", val_data.dim().0);

        // Créer l'optimiseur
        let mut optimizer: Box<dyn Optimizer> = match &self.training_config.optimizer_type {
            super::config::OptimizerType::Adam { beta1, beta2, epsilon } => {
                Box::new(Adam::new(
                    self.training_config.learning_rate,
                    *beta1,
                    *beta2,
                    *epsilon,
                ))
            }
            super::config::OptimizerType::SGD { momentum } => {
                Box::new(SGD::new(self.training_config.learning_rate, *momentum))
            }
        };

        for epoch in 0..self.training_config.num_epochs {
            self.state.current_epoch = epoch;
            
            println!("\n📅 Époque {}/{}", epoch + 1, self.training_config.num_epochs);
            
            // Phase d'entraînement
            let (train_loss, train_acc) = self.train_epoch(
                &train_data, 
                &train_labels, 
                optimizer.as_mut()
            );
            
            // Phase de validation
            let (val_loss, val_acc) = self.validate(&val_data, &val_labels);
            
            // Mise à jour de l'historique
            self.history.update(train_loss, val_loss, train_acc, val_acc);
            
            // Affichage
            println!("📊 Loss: {:.4} (train) {:.4} (val) | Acc: {:.2}% (train) {:.2}% (val)",
                     train_loss, val_loss, train_acc * 100.0, val_acc * 100.0);
            
            // Early stopping
            if self.check_early_stopping(val_loss) {
                println!("🛑 Arrêt précoce à l'époque {}", epoch + 1);
                break;
            }
        }

        self.state.is_trained = true;
        println!("\n✅ ENTRAÎNEMENT TERMINÉ");
        self
    }

    /// Entraînement sur une époque
    fn train_epoch(
        &mut self,
        data: &Array4<f32>,
        labels: &Array4<f32>,
        optimizer: &mut dyn Optimizer,
    ) -> (f32, f32) {
        let num_samples = data.dim().0;
        let batch_size = self.training_config.batch_size;
        let num_batches = (num_samples + batch_size - 1) / batch_size;
        
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_samples);
            
            let batch_data = data.slice(s![start..end, .., .., ..]).to_owned();
            let batch_labels = labels.slice(s![start..end, .., .., ..]).to_owned();
            
            // Forward pass
            let output = self.forward(&batch_data.view());
            
            // Calculer la perte
            let loss = self.loss_fn.compute_total(
                &output.view(),
                &batch_labels.view(),
                None,
                None,
            );
            
            // Calculer l'accuracy
            let metrics = Metrics::compute(&output.view(), &batch_labels.view());
            
            // Backward pass
            let grad_loss = self.loss_fn.gradient_capsules(&output.view(), &batch_labels.view());
            self.backward(&grad_loss.view());
            
            // Mise à jour des poids
            self.update_weights(optimizer);
            
            total_loss += loss;
            total_acc += metrics.accuracy;
            
            if (batch_idx + 1) % 10 == 0 {
                println!("   Batch {}/{} - Loss: {:.4}, Acc: {:.2}%",
                         batch_idx + 1, num_batches, loss, metrics.accuracy * 100.0);
            }
        }
        
        (total_loss / num_batches as f32, total_acc / num_batches as f32)
    }

    /// Validation
    pub fn validate(&mut self, data: &Array4<f32>, labels: &Array4<f32>) -> (f32, f32) {
        let output = self.forward(&data.view());
        
        let loss = self.loss_fn.compute_total(
            &output.view(),
            &labels.view(),
            None,
            None,
        );
        
        let metrics = Metrics::compute(&output.view(), &labels.view());
        
        (loss, metrics.accuracy)
    }

    /// Prédiction
    pub fn predict(&mut self, data: &Array4<f32>) -> Vec<usize> {
        let output = self.forward(&data.view());
        let (batch_size, num_classes, _, dim) = output.dim();
        
        let mut predictions = Vec::with_capacity(batch_size);
        
        for b in 0..batch_size {
            let mut max_norm = 0.0;
            let mut pred_class = 0;
            
            for c in 0..num_classes {
                let mut norm_sq = 0.0;
                for d in 0..dim {
                    let val = output[[b, c, 0, d]];
                    norm_sq += val * val;
                }
                let norm = norm_sq.sqrt();
                
                if norm > max_norm {
                    max_norm = norm;
                    pred_class = c;
                }
            }
            
            predictions.push(pred_class);
        }
        
        predictions
    }

    /// Early stopping
    fn check_early_stopping(&mut self, current_loss: f32) -> bool {
        if self.training_config.early_stopping_patience == 0 {
            return false;
        }

        if current_loss < self.state.best_loss {
            self.state.best_loss = current_loss;
            self.state.early_stopping_counter = 0;
        } else {
            self.state.early_stopping_counter += 1;
        }

        self.state.early_stopping_counter >= self.training_config.early_stopping_patience
    }

    /// Diagnostic rapide
    pub fn diagnostic(&mut self) {
        println!("🔍 DIAGNOSTIC RAPIDE");
        println!("   Couches: {}", self.layers.len());
        println!("   Input shape: {:?}", self.network_config.input_shape);
        
        let (c, h, w) = self.network_config.input_shape;
        let test_input = Array4::zeros((1, c, h, w));
        
        let output = self.forward(&test_input.view());
        println!("   Output shape: {:?}", output.dim());
        println!("✅ Modèle opérationnel");
    }
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_accuracy: Vec::new(),
            val_accuracy: Vec::new(),
        }
    }

    pub fn update(&mut self, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) {
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);
        self.train_accuracy.push(train_acc);
        self.val_accuracy.push(val_acc);
    }
}

impl ModelState {
    pub fn new() -> Self {
        Self {
            is_trained: false,
            best_loss: f32::INFINITY,
            current_epoch: 0,
            early_stopping_counter: 0,
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self::new()
    }
}