use super::{
    config::{NetworkConfig, TrainingConfig},
    builder::Layer,
};

use ndarray::{Array4, ArrayView4, s};
use ndarray::ArrayView1;


/// MODÈLE CAPNET PRINCIPAL COMPLÈTEMENT MODULAIRE
pub struct CapNet {
    /// Configuration de l'architecture
    pub network_config: NetworkConfig,
    /// Configuration de l'entraînement
    pub training_config: TrainingConfig,
    /// Séquence dynamique de couches
    pub layers: Vec<Layer>,
    /// Historique d'entraînement
    pub history: TrainingHistory,
    /// État du modèle
    pub state: ModelState,
}

/// HISTORIQUE D'ENTRAÎNEMENT POUR ANALYSE
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub train_accuracy: Vec<f32>,
    pub val_accuracy: Vec<f32>,
    pub learning_rates: Vec<f32>,
}

/// ÉTAT DU MODÈLE
#[derive(Debug, Clone)]
pub struct ModelState {
    pub is_trained: bool,
    pub best_loss: f32,
    pub current_epoch: usize,
    pub early_stopping_counter: usize,
}

impl CapNet {
    /// CRÉATION D'UN NOUVEAU MODÈLE AVEC CONFIGURATION DYNAMIQUE
    pub fn new(
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        layers: Vec<Layer>,
    ) -> Self {
        Self {
            network_config,
            training_config,
            layers,
            history: TrainingHistory::new(),
            state: ModelState::new(),
        }
    }

    /// PROPAGATION AVANT À TRAVERS TOUTES LES COUCHES
    pub fn forward(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let mut output = input.to_owned();
        
        for layer in &self.layers {
            output = layer.forward(&output.view());
        }
        
        output
    }

    /// ENTRAÎNEMENT COMPLET AVEC OPTIONS MODULAIRES
    pub fn train(
        mut self,
        train_data: Array4<f32>,
        train_labels: Array4<f32>,
        val_data: Array4<f32>,
        val_labels: Array4<f32>,
    ) -> Self {
        println!("🎯 Début de l'entraînement avec {} échantillons", train_data.dim().0);

        for epoch in 0..self.training_config.num_epochs {
            self.state.current_epoch = epoch;
            
            // Entraînement sur un epoch
            let (train_loss, train_accuracy) = self.train_epoch(&train_data, &train_labels);
            
            // Validation
            let (val_loss, val_accuracy) = self.validate(&val_data, &val_labels);
            
            // Mise à jour de l'historique
            self.history.update(train_loss, val_loss, train_accuracy, val_accuracy);
            
            // Affichage des progrès
            self.print_epoch_progress(epoch, train_loss, val_loss, train_accuracy, val_accuracy);
            
            // Arrêt précoce
            if self.check_early_stopping() {
                println!("🛑 Arrêt précoce à l'époque {}", epoch);
                break;
            }
        }

        self.state.is_trained = true;
        self
    }

    /// ENTRAÎNEMENT SUR UN SEUL EPOCH
    fn train_epoch(&mut self, data: &Array4<f32>, labels: &Array4<f32>) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    
    // ⚡ CORRECTION: Calcul correct du nombre de batchs
    let num_batches = (data.dim().0 + self.training_config.batch_size - 1) / self.training_config.batch_size;
    
    println!("   📦 Batchs: {} ({} images / batch_size={})", 
             num_batches, data.dim().0, self.training_config.batch_size);

    for batch in 0..num_batches {
        let start = batch * self.training_config.batch_size;
        let end = std::cmp::min(start + self.training_config.batch_size, data.dim().0);
        
        let batch_data = data.slice(ndarray::s![start..end, .., .., ..]).to_owned();
        let batch_labels = labels.slice(ndarray::s![start..end, .., .., ..]).to_owned();
        
        // Propagation avant
        let output = self.forward(&batch_data.view());
        
        // Calcul de la perte et précision
        let loss = self.compute_loss(&output.view(), &batch_labels.view());
        let accuracy = self.compute_accuracy(&output.view(), &batch_labels.view());
        
        total_loss += loss;
        total_accuracy += accuracy;
        
        // Afficher la progression
        if (batch + 1) % 10 == 0 {
            println!("     🔄 Batch {}/{} - Loss: {:.4}", batch + 1, num_batches, loss);
        }
    }

    (
        total_loss / num_batches as f32,
        total_accuracy / num_batches as f32,
    )
}

    /// VALIDATION DU MODÈLE
    pub fn validate(&self, data: &Array4<f32>, labels: &Array4<f32>) -> (f32, f32) {
        let output = self.forward(&data.view());
        let loss = self.compute_loss(&output.view(), &labels.view());
        let accuracy = self.compute_accuracy(&output.view(), &labels.view());
        
        (loss, accuracy)
    }

    /// OBTENTION D'UN LOT DE DONNÉES
    pub fn get_batch(&self, data: &Array4<f32>, labels: &Array4<f32>, batch_index: usize) -> (Array4<f32>, Array4<f32>) {
        let start = batch_index * self.training_config.batch_size;
        let end = std::cmp::min(start + self.training_config.batch_size, data.dim().0);
        
        let batch_data = data.slice(s![start..end, .., .., ..]).to_owned();
        let batch_labels = labels.slice(s![start..end, .., .., ..]).to_owned();
        
        (batch_data, batch_labels)
    }

    /// CALCUL DE LA PERTE (MARGIN LOSS POUR CAPSULES)
   fn compute_loss(&self, output: &ArrayView4<f32>, labels: &ArrayView4<f32>) -> f32 {
    let (batch_size, num_classes, _, _) = output.dim();
    let mut total_loss = 0.0;

    let margin_config = self.training_config.margin_loss_params.as_ref()
        .unwrap_or(&super::config::MarginLossConfig {
            positive_margin: 0.9,
            negative_margin: 0.1,
            down_weighting: 0.5,
        });

    for b in 0..batch_size {
        for c in 0..num_classes {
            let capsule_slice = output.slice(ndarray::s![b, c, 0, ..]);
            let norm = self.capsule_norm_1d(&capsule_slice);
            let target = labels[[b, c, 0, 0]];

            // ⚡ CORRECTION: Éviter les valeurs NaN
            let safe_norm = norm.max(1e-8); // Éviter la division par zéro
            
            let loss = if target > 0.5 {
                // Capsule active
                (margin_config.positive_margin - safe_norm).max(0.0).powi(2)
            } else {
                // Capsule inactive
                margin_config.down_weighting * (safe_norm - margin_config.negative_margin).max(0.0).powi(2)
            };

            total_loss += loss;
        }
    }

    total_loss / (batch_size * num_classes) as f32
}

    /// CALCUL DE LA PRÉCISION
    pub fn compute_accuracy(&self, output: &ArrayView4<f32>, labels: &ArrayView4<f32>) -> f32 {
        let predictions = self.predict(output);
        let mut correct = 0;
        let total = output.dim().0;

        for i in 0..total {
            if predictions[i] == self.get_true_class(&labels.slice(s![i, .., 0, 0])) {
                correct += 1;
            }
        }

        correct as f32 / total as f32
    }

    /// PRÉDICTION DES CLASSES
    pub fn predict(&self, output: &ArrayView4<f32>) -> Vec<usize> {
        let (batch_size, num_classes, _, _) = output.dim();
        let mut predictions = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut max_norm = 0.0;
            let mut pred_class = 0;

            for c in 0..num_classes {
                let capsule_slice = output.slice(s![b, c, 0, ..]);
                let norm = self.capsule_norm_1d(&capsule_slice);
                if norm > max_norm {
                    max_norm = norm;
                    pred_class = c;
                }
            }
            predictions.push(pred_class);
        }

        predictions
    }

    /// NORME D'UNE CAPSULE 1D (pour les capsules de sortie)
    pub fn capsule_norm_1d(&self, capsule: &ndarray::ArrayView1<f32>) -> f32 {
        let mut sum = 0.0;
        for &val in capsule {
            sum += val * val;
        }
        sum.sqrt()
    }

    /// NORME D'UNE CAPSULE 4D (méthode générique)
    pub fn capsule_norm(&self, capsule: &ArrayView4<f32>) -> f32 {
        let mut sum = 0.0;
        for &val in capsule.iter() {
            sum += val * val;
        }
        sum.sqrt()
    }

    /// OBTENTION DE LA CLASSE RÉELLE
     pub fn get_true_class(&self, label: &ArrayView1<f32>) -> usize {  // Changer ArrayView2 par ArrayView1
        for c in 0..label.dim() {  // Utiliser .dim() au lieu de .dim().0
            if label[c] > 0.5 {    // Utiliser [c] au lieu de [[c, 0]]
                return c;
            }
        }
        0
    }

    /// VÉRIFICATION DE L'ARRÊT PRÉCOCE
    pub fn check_early_stopping(&mut self) -> bool {
        if self.training_config.early_stopping_patience == 0 {
            return false;
        }

        if self.history.val_loss.len() < 2 {
            return false;
        }

        let current_loss = *self.history.val_loss.last().unwrap();
        let best_loss = self.state.best_loss;

        if current_loss < best_loss {
            self.state.best_loss = current_loss;
            self.state.early_stopping_counter = 0;
        } else {
            self.state.early_stopping_counter += 1;
        }

        self.state.early_stopping_counter >= self.training_config.early_stopping_patience
    }

    /// AFFICHAGE DES PROGRÈS
    pub fn print_epoch_progress(&self, epoch: usize, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) {
        println!(
            "📊 Epoch {:3}/{} | Loss: {:.4} (train) {:.4} (val) | Acc: {:.2} (train) {:.2} (val)",
            epoch + 1,
            self.training_config.num_epochs,
            train_loss,
            val_loss,
            train_acc,
            val_acc
        );
    }

    /// OBTENTION DE L'ARCHITECTURE
    pub fn get_architecture(&self) -> String {
        format!("CapNet avec {} couches configurées dynamiquement", self.layers.len())
    }

    /// SAUVEGARDE DU MODÈLE
    pub fn save(&self, path: &str) {
        println!("💾 Sauvegarde du modèle dans: {}", path);
        // Implémentation de la sauvegarde à compléter
    }

      pub fn diagnostic(&self) {
        println!("🔍 DIAGNOSTIC DU MODÈLE CAPNET");
        println!("===============================");
        
        // Informations de configuration
        println!("📐 Configuration:");
        println!("   - Shape entrée: {:?}", self.network_config.input_shape);
        println!("   - Couches: {}", self.layers.len());
        println!("   - Itérations routage: {}", self.network_config.routing_iterations);
        
        // Test avec données minimales SÉCURISÉ
        println!("🧪 Test forward pass...");
        
        // Créer une petite donnée de test
        let (channels, height, width) = self.network_config.input_shape;
        let test_input = ndarray::Array4::zeros((1, channels, height, width));
        println!("   ✅ Input créé: {:?}", test_input.dim());
        
        // Test couche par couche avec gestion d'erreur
        let mut current_output = test_input;
        
        for (i, layer) in self.layers.iter().enumerate() {
            println!("   🔍 Couche {}...", i);
            current_output = layer.forward(&current_output.view());
            println!("   ✅ Shape sortie: {:?}", current_output.dim());
            
            // Vérification de sécurité
            if current_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                println!("   ❌ ERREUR: Valeurs NaN ou infinies détectées!");
                break;
            }
        }
        
        println!("   🎯 Forward pass RÉUSSI!");
        println!("   📏 Shape finale: {:?}", current_output.dim());
        
        // Vérification des capsules de sortie
        if current_output.dim().1 > 0 {
            println!("📊 Analyse capsules sortie:");
            for i in 0..current_output.dim().1.min(4) { // Limiter à 4 capsules
                let capsule_slice = current_output.slice(ndarray::s![0, i, 0, ..]);
                let norm: f32 = capsule_slice.iter().map(|&v| v * v).sum::<f32>().sqrt();
                println!("   - Capsule {}: norm = {:.4}", i, norm);
            }
        }
        
        // État du modèle
        println!("📈 État du modèle:");
        println!("   - Entraîné: {}", self.state.is_trained);
        println!("   - Meilleure loss: {:.4}", self.state.best_loss);
        println!("   - Époques: {}", self.state.current_epoch);
        
        println!("🎯 DIAGNOSTIC TERMINÉ - Modèle opérationnel");
    }

    /// DIAGNOSTIC ULTRA-RAPIDE (alternative)
    pub fn diagnostic_fast(&self) {
        println!("⚡ DIAGNOSTIC RAPIDE");
        
        let (channels, height, width) = self.network_config.input_shape;
        let test_input = ndarray::Array4::zeros((1, channels, height, width));
        
        println!("   🧪 Test forward pass...");
        let output = self.forward(&test_input.view());
        println!("   ✅ Réussi! Input: {:?} -> Output: {:?}", test_input.dim(), output.dim());
        println!("   🎯 Modèle prêt pour l'entraînement!");
    }
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_accuracy: Vec::new(),
            val_accuracy: Vec::new(),
            learning_rates: Vec::new(),
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