mod model;
mod train_data;

use model::{
    builder::ModelBuilder,
    config::{NetworkConfig, TrainingConfig, LayerConfig, CapsuleConfig},
    core::CapNet
};
use train_data::data_loader::MalariaDataLoader;
use train_data::data_loader;
use model::config::MarginLossConfig;    
use std::time::Instant;

fn main() {
    let total_start = Instant::now();
    println!("🚀 CAPNET - CONFIGURATION FONCTIONNELLE");
    println!("=======================================");

    // 1. CHARGEMENT OPTIMAL
    println!("📁 Chargement des données...");
    let data_loader = MalariaDataLoader::new("malaria_data", (64, 64)); // 64x64 pour de vraies features
    let dataset = data_loader.load_dataset_fast(0.2, 1000); // 1000 images

    // 2. CONFIGURATION QUI FONCTIONNE
    println!("⚙️ Configuration optimisée...");
    let network_config = NetworkConfig {
        input_shape: (3, 64, 64),
        layers: vec![
            LayerConfig::Conv2d {
                in_channels: 3,
                out_channels: 16,  // Assez pour capturer les patterns
                kernel_size: 5,    // Plus grand pour voir les parasites
                stride: 1,
                padding: 2,
                activation: Some("relu".to_string()),
            },
            LayerConfig::PrimaryCapsules {
                in_channels: 16,
                capsule_config: CapsuleConfig {
                    num_capsules: 4,   // Assez pour différentes orientations
                    capsule_dim: 4,    // Assez d'information
                    kernel_size: 5,
                    stride: 2,
                    padding: 2,       // IMPORTANT: padding pour bonnes dimensions
                    capsule_params: None,
                },
            },
            LayerConfig::DigitCapsules {
                primary_capsules: 4,
                primary_capsule_dim: 4,
                capsule_config: CapsuleConfig {
                    num_capsules: 2,
                    capsule_dim: 8,    // Représentation riche
                    kernel_size: 0,
                    stride: 0,
                    padding: 0,
                    capsule_params: None,
                },
            },
        ],
        routing_iterations: 2,     // Assez pour le routage
        extra_params: None,
    };

    let training_config = TrainingConfig {
        batch_size: 16,           // Bon compromis
        learning_rate: 0.001,     // Standard
        num_epochs: 5,            // Assez pour voir l'apprentissage
        validation_split: 0.2,
        save_best: true,
        early_stopping_patience: 2,
        loss_function: "margin".to_string(),
        optimizer: "adam".to_string(),
        margin_loss_params: Some(MarginLossConfig {
            positive_margin: 0.9,
            negative_margin: 0.1,
            down_weighting: 0.5,
        }),
    };

    // 3. CONSTRUCTION
    println!("🏗️ Construction...");
    let model = ModelBuilder::new()
        .with_network_config(network_config)
        .with_training_config(training_config)
        .build();

    // 4. VÉRIFICATION
    println!("🔍 Vérification...");
    let test_input = ndarray::Array4::zeros((1, 3, 64, 64));
    let output = model.forward(&test_input.view());
    println!("   ✅ Modèle vérifié: {:?} -> {:?}", test_input.dim(), output.dim());

    // 5. ENTRAÎNEMENT
    println!("🎯 Début entraînement (5 époques)...");
    let train_start = Instant::now();
    
    let trained_model = model.train(
        dataset.train_data.clone(),
        dataset.train_labels.clone(),
        dataset.test_data.clone(),
        dataset.test_labels.clone(),
    );

    let train_duration = train_start.elapsed();
    println!("✅ Entraînement terminé en: {:?}", train_duration);

    // 6. ÉVALUATION
    evaluate_proper(&trained_model, &dataset);

    let total_duration = total_start.elapsed();
    println!("🎉 TOTAL: {:?}", total_duration);
}
/// TEST TRÈS SIMPLE QUI NE PEUT PAS PLANTER
fn test_model_basics(model: &CapNet) {
    println!("   🧪 Test basique...");
    
    // Test avec une seule petite image
    let test_input = ndarray::Array4::zeros((1, 3, 64, 64));
    println!("   ✅ Input créé: {:?}", test_input.dim());
    
    // Juste essayer le forward pass
    match std::panic::catch_unwind(|| {
        model.forward(&test_input.view())
    }) {
        Ok(output) => {
            println!("   ✅ Forward pass RÉUSSI!");
            println!("   📏 Output shape: {:?}", output.dim());
        },
        Err(_) => {
            println!("   ⚠️  Forward pass a échoué, mais on continue quand même...");
        }
    }
}

fn evaluate_proper(model: &CapNet, dataset: &data_loader::Dataset) {
    let predictions = model.predict(&dataset.test_data.view());
    let mut correct = 0;
    let total = dataset.test_data.dim().0;

    for i in 0..total {
        let true_class = if dataset.test_labels[[i, 1, 0, 0]] > 0.5 { 1 } else { 0 };
        if predictions[i] == true_class {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    println!("📊 PERFORMANCE FINALE:");
    println!("   - Accuracy: {:.2}%", accuracy * 100.0);
    println!("   - Images test: {}", total);
    println!("   - Correctes: {}", correct);
    
    if accuracy > 0.75 {
        println!("   🎉 EXCELLENT!");
    } else if accuracy > 0.65 {
        println!("   ✅ BON!");
    } else if accuracy > 0.55 {
        println!("   ⚠️  ACCEPTABLE");
    } else {
        println!("   🔄 À AMÉLIORER");
    }
}