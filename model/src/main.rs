mod model;
mod train_data;

use model::{
    config::{NetworkConfig, TrainingConfig, LayerConfig, CapsuleConfig, Activation, OptimizerType, LossConfig},
    ModelBuilder,
};
use train_data::data_loader::MalariaDataLoader;

fn main() {
    println!("🚀 CAPSNET - DÉTECTION DU PALUDISME");
    println!("===================================\n");

    // Configuration du réseau
    let network_config = NetworkConfig {
        input_shape: (3, 64, 64),
        layers: vec![
            // Couche convolutive 1
            LayerConfig::Conv2d {
                in_channels: 3,
                out_channels: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                activation: Activation::ReLU,
            },
            
            // Couche convolutive 2
            LayerConfig::Conv2d {
                in_channels: 64,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                activation: Activation::ReLU,
            },
            
            // Capsules primaires
            LayerConfig::PrimaryCapsules {
                in_channels: 128,
                capsule_config: CapsuleConfig {
                    num_capsules: 32,
                    capsule_dim: 8,
                    kernel_size: 9,
                    stride: 2,
                    padding: 0,
                },
            },
            
            // Capsules de sortie (2 classes: infecté / sain)
            LayerConfig::DigitCapsules {
                input_capsules: 32,
                input_capsule_dim: 8,
                output_capsules: 2,
                output_capsule_dim: 16,
            },
        ],
        routing_iterations: 3,
        use_reconstruction: false,
        extra_params: None,
    };

    // Configuration de l'entraînement
    let training_config = TrainingConfig {
        batch_size: 16,
        learning_rate: 0.001,
        num_epochs: 30,
        validation_split: 0.2,
        save_best: true,
        early_stopping_patience: 5,
        optimizer_type: OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        loss_config: LossConfig {
            positive_margin: 0.9,
            negative_margin: 0.1,
            down_weighting: 0.5,
            reconstruction_weight: 0.0005,
        },
        lr_schedule: None,
    };

    // Construction du modèle
    println!("🏗️  Construction du modèle...");
    let mut model = ModelBuilder::new()
        .with_network_config(network_config)
        .with_training_config(training_config)
        .build()
        .expect("Erreur lors de la construction du modèle");

    println!("✅ Modèle construit avec succès\n");

    // Diagnostic
    model.diagnostic();
    println!();

    // Chargement des données
    println!("📁 Chargement des données...");
    let data_path = "malaria_data";
    let loader = MalariaDataLoader::new(data_path, (64, 64));
    
    // Charger un petit échantillon pour le test
    let dataset = loader.load_dataset_fast(0.2, 1000);
    
    println!("✅ Données chargées:");
    println!("   Train: {} échantillons", dataset.train_data.dim().0);
    println!("   Test: {} échantillons\n", dataset.test_data.dim().0);

    // Entraînement
    println!("🎯 Début de l'entraînement...\n");
    let trained_model = model.train(
        dataset.train_data,
        dataset.train_labels,
        dataset.test_data,
        dataset.test_labels,
    );

    println!("\n🎉 ENTRAÎNEMENT TERMINÉ !");
    println!("   Meilleure loss validation: {:.4}", trained_model.state.best_loss);
    
    // Sauvegarder le modèle
    println!("\n💾 Sauvegarde du modèle...");
    // trained_model.save("models/capsnet_malaria.bin");
    println!("✅ Modèle sauvegardé");
}