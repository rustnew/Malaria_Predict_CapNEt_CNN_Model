use ndarray::{Array4, Array3};
use image::ImageReader;
use std::path::{Path, PathBuf};
use std::fs;
use rand::seq::SliceRandom;
use std::time::Instant;

/// Chargeur ULTRA RAPIDE
pub struct MalariaDataLoader {
    parasitized_path: PathBuf,
    uninfected_path: PathBuf,
    image_size: (usize, usize),
}

impl MalariaDataLoader {
    pub fn new(data_path: &str, image_size: (usize, usize)) -> Self {
        Self {
            parasitized_path: PathBuf::from(data_path).join("Parasitized"),
            uninfected_path: PathBuf::from(data_path).join("Uninfected"),
            image_size,
        }
    }

    /// Charge un ÉCHANTILLON RAPIDE
    pub fn load_dataset_fast(&self, test_split: f32, max_samples: usize) -> Dataset {
        println!("📁 Chargement RAPIDE des données...");
        let start = Instant::now();

        let parasitized = self.load_images_from_dir_fast(&self.parasitized_path, 1.0, max_samples);
        let uninfected = self.load_images_from_dir_fast(&self.uninfected_path, 0.0, max_samples);

        println!("📊 Échantillon chargé:");
        println!("   - Infectées: {}", parasitized.len());
        println!("   - Saines: {}", uninfected.len());
        println!("   - Total: {}", parasitized.len() + uninfected.len());

        // Mélanger
        let mut all_data = parasitized;
        all_data.extend(uninfected);
        let mut rng = rand::rng();
        all_data.shuffle(&mut rng);

        // Split
        let split_index = (all_data.len() as f32 * (1.0 - test_split)) as usize;
        let (train_data, test_data) = all_data.split_at(split_index);

        println!("🎯 Split: Train={}, Test={}", train_data.len(), test_data.len());

        let dataset = self.prepare_arrays_fast(train_data, test_data);
        
        println!("⏱️  Chargé en: {:?}", start.elapsed());
        dataset
    }

    fn load_images_from_dir_fast(&self, dir_path: &Path, label: f32, max_samples: usize) -> Vec<(Array3<f32>, f32)> {
        let mut images = Vec::new();
        
        if let Ok(entries) = fs::read_dir(dir_path) {
            for (i, entry) in entries.flatten().enumerate().take(max_samples) {
                if i % 500 == 0 {
                    println!("     📸 {} images chargées...", i);
                }
                
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "png" || ext == "jpg" || ext == "jpeg" {
                        if let Some(image_array) = self.load_image_fast(&path) {
                            images.push((image_array, label));
                        }
                    }
                }
            }
        }
        
        images
    }

    fn load_image_fast(&self, image_path: &Path) -> Option<Array3<f32>> {
        let img = ImageReader::open(image_path).ok()?.decode().ok()?;

        // Redimensionnement RAPIDE
        let resized = img.resize_exact(
            self.image_size.0 as u32,
            self.image_size.1 as u32,
            image::imageops::FilterType::Triangle, // Équilibre vitesse/qualité
        );

        let rgb = resized.to_rgb32f();
        let (width, height) = rgb.dimensions();

        let mut array = Array3::zeros((3, height as usize, width as usize));
        
        for (x, y, pixel) in rgb.enumerate_pixels() {
            array[[0, y as usize, x as usize]] = pixel.0[0];
            array[[1, y as usize, x as usize]] = pixel.0[1];
            array[[2, y as usize, x as usize]] = pixel.0[2];
        }

        Some(array)
    }

    fn prepare_arrays_fast(&self, train_data: &[(Array3<f32>, f32)], test_data: &[(Array3<f32>, f32)]) -> Dataset {
        let train_count = train_data.len();
        let test_count = test_data.len();
        let (channels, height, width) = (3, self.image_size.1, self.image_size.0);

        let mut train_images = Array4::zeros((train_count, channels, height, width));
        let mut train_labels = Array4::zeros((train_count, 2, 1, 1));

        // Chargement PARALLÉLISÉ (conceptuel)
        for (i, (image, label)) in train_data.iter().enumerate() {
            train_images.slice_mut(ndarray::s![i, .., .., ..]).assign(image);
            
            if *label == 1.0 {
                train_labels[[i, 1, 0, 0]] = 1.0;
            } else {
                train_labels[[i, 0, 0, 0]] = 1.0;
            }
        }

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