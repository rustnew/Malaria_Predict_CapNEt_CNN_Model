use ndarray::{Array4, Array3};
use image::ImageReader;
use std::path::{Path, PathBuf};
use std::fs;
use rand::seq::SliceRandom;
use std::time::Instant;

/// Chargeur de données optimisé
pub struct MalariaDataLoader {
    parasitized_path: PathBuf,
    uninfected_path: PathBuf,
    image_size: (usize, usize),
}

impl MalariaDataLoader {
    
    pub fn new(data_path: &str, image_size: (usize, usize)) -> Self {
        let parasitized_path = PathBuf::from(data_path).join("Parasitized");
        let uninfected_path = PathBuf::from(data_path).join("Uninfected");
        
        println!("📂 Chemins de données:");
        println!("   Parasitized: {:?}", parasitized_path);
        println!("   Uninfected: {:?}", uninfected_path);
        
        // Vérifier que les dossiers existent
        if !parasitized_path.exists() {
            eprintln!("⚠️  ATTENTION: Le dossier Parasitized n'existe pas!");
        }
        if !uninfected_path.exists() {
            eprintln!("⚠️  ATTENTION: Le dossier Uninfected n'existe pas!");
        }
        
        Self {
            parasitized_path,
            uninfected_path,
            image_size,
        }
    }

    /// Charge un échantillon rapide de données
    pub fn load_dataset_fast(&self, test_split: f32, max_samples: usize) -> Dataset {
        println!("📁 Chargement RAPIDE des données...");
        let start = Instant::now();

        // Charger les images parasitées
        println!("🦠 Chargement images parasitées...");
        let parasitized = self.load_images_from_dir_fast(
            &self.parasitized_path, 
            1.0, 
            max_samples / 2
        );

        // Charger les images saines
        println!("✅ Chargement images saines...");
        let uninfected = self.load_images_from_dir_fast(
            &self.uninfected_path, 
            0.0, 
            max_samples / 2
        );

        println!("📊 Échantillon chargé:");
        println!("   - Infectées: {}", parasitized.len());
        println!("   - Saines: {}", uninfected.len());
        println!("   - Total: {}", parasitized.len() + uninfected.len());

        if parasitized.is_empty() && uninfected.is_empty() {
            eprintln!("❌ ERREUR: Aucune image chargée!");
            eprintln!("   Vérifiez que les dossiers contiennent des images PNG/JPG");
        }

        // Mélanger les données
        let mut all_data = parasitized;
        all_data.extend(uninfected);
        
        if all_data.is_empty() {
            eprintln!("⚠️  Dataset vide - création d'un dataset factice pour test");
            return Dataset::empty();
        }
        
        let mut rng = rand::thread_rng();
        all_data.shuffle(&mut rng);

        // Split train/test
        let split_index = (all_data.len() as f32 * (1.0 - test_split)) as usize;
        let (train_data, test_data) = all_data.split_at(split_index);

        println!("🎯 Split: Train={}, Test={}", train_data.len(), test_data.len());

        let dataset = self.prepare_arrays_fast(train_data, test_data);
        
        println!("⏱️  Chargé en: {:?}", start.elapsed());
        dataset
    }

    fn load_images_from_dir_fast(
        &self, 
        dir_path: &Path, 
        label: f32, 
        max_samples: usize
    ) -> Vec<(Array3<f32>, f32)> {
        let mut images = Vec::new();
        
        // Vérifier si le dossier existe
        if !dir_path.exists() {
            eprintln!("❌ Dossier n'existe pas: {:?}", dir_path);
            return images;
        }
        
        // Lire le contenu du dossier
        let entries = match fs::read_dir(dir_path) {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!("❌ Erreur lecture dossier {:?}: {}", dir_path, e);
                return images;
            }
        };
        
        let mut count = 0;
        let mut loaded = 0;
        
        for entry in entries.flatten() {
            if loaded >= max_samples {
                break;
            }
            
            count += 1;
            let path = entry.path();
            
            // Vérifier l'extension
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "png" || ext_str == "jpg" || ext_str == "jpeg" {
                    if let Some(image_array) = self.load_image_fast(&path) {
                        images.push((image_array, label));
                        loaded += 1;
                        
                        if loaded % 100 == 0 {
                            println!("     📸 {} images chargées...", loaded);
                        }
                    } else {
                        if count <= 10 {
                            eprintln!("     ⚠️  Échec chargement: {:?}", path);
                        }
                    }
                }
            }
        }
        
        println!("     ✅ {} images valides sur {} fichiers analysés", loaded, count);
        images
    }

    fn load_image_fast(&self, image_path: &Path) -> Option<Array3<f32>> {
        // Charger l'image
        let img = match ImageReader::open(image_path) {
            Ok(reader) => match reader.decode() {
                Ok(img) => img,
                Err(_) => return None,
            },
            Err(_) => return None,
        };

        // Redimensionner
        let resized = img.resize_exact(
            self.image_size.0 as u32,
            self.image_size.1 as u32,
            image::imageops::FilterType::Triangle,
        );

        let rgb = resized.to_rgb32f();
        let (width, height) = rgb.dimensions();

        // Convertir en Array3
        let mut array = Array3::zeros((3, height as usize, width as usize));
        
        for (x, y, pixel) in rgb.enumerate_pixels() {
            array[[0, y as usize, x as usize]] = pixel.0[0];
            array[[1, y as usize, x as usize]] = pixel.0[1];
            array[[2, y as usize, x as usize]] = pixel.0[2];
        }

        Some(array)
    }

    fn prepare_arrays_fast(
        &self, 
        train_data: &[(Array3<f32>, f32)], 
        test_data: &[(Array3<f32>, f32)]
    ) -> Dataset {
        let train_count = train_data.len();
        let test_count = test_data.len();
        let (channels, height, width) = (3, self.image_size.1, self.image_size.0);

        // Préparer les arrays d'entraînement
        let mut train_images = Array4::zeros((train_count, channels, height, width));
        let mut train_labels = Array4::zeros((train_count, 2, 1, 1));

        for (i, (image, label)) in train_data.iter().enumerate() {
            train_images.slice_mut(ndarray::s![i, .., .., ..]).assign(image);
            
            if *label == 1.0 {
                train_labels[[i, 1, 0, 0]] = 1.0; // Parasitized
            } else {
                train_labels[[i, 0, 0, 0]] = 1.0; // Uninfected
            }
        }

        // Préparer les arrays de test
        let mut test_images = Array4::zeros((test_count, channels, height, width));
        let mut test_labels = Array4::zeros((test_count, 2, 1, 1));

        for (i, (image, label)) in test_data.iter().enumerate() {
            test_images.slice_mut(ndarray::s![i, .., .., ..]).assign(image);
            
            if *label == 1.0 {
                test_labels[[i, 1, 0, 0]] = 1.0;
            } else {
                test_labels[[i, 0, 0, 0]] = 1.0;
            }
        }

        Dataset {
            train_data: train_images,
            train_labels,
            test_data: test_images,
            test_labels,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub train_data: Array4<f32>,
    pub train_labels: Array4<f32>,
    pub test_data: Array4<f32>,
    pub test_labels: Array4<f32>,
}

impl Dataset {
    /// Crée un dataset vide pour les tests
    pub fn empty() -> Self {
        Self {
            train_data: Array4::zeros((0, 3, 64, 64)),
            train_labels: Array4::zeros((0, 2, 1, 1)),
            test_data: Array4::zeros((0, 3, 64, 64)),
            test_labels: Array4::zeros((0, 2, 1, 1)),
        }
    }
}