use ndarray::Array4;
use std::collections::HashMap;

/// Trait pour les optimiseurs
pub trait Optimizer: Send + Sync {
    fn step(&mut self, param_id: &str, param: &mut Array4<f32>, grad: &Array4<f32>);
    fn zero_grad(&mut self);
    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

/// Optimiseur SGD avec momentum
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocity: HashMap<String, Array4<f32>>,
}

impl SGD {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param_id: &str, param: &mut Array4<f32>, grad: &Array4<f32>) {
        // Initialiser la vélocité si nécessaire
        let velocity = self.velocity
            .entry(param_id.to_string())
            .or_insert_with(|| Array4::zeros(param.dim()));
        
        // Mise à jour de la vélocité: v = momentum * v - lr * grad
        *velocity = &*velocity * self.momentum - grad * self.learning_rate;
        
        // Mise à jour des paramètres: param += v
        *param += &*velocity;
    }
    
    fn zero_grad(&mut self) {
        // SGD ne nécessite pas de réinitialisation des gradients accumulés
    }
    
    fn get_lr(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Optimiseur Adam
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: usize,
    
    // Moments du premier ordre (mean)
    m: HashMap<String, Array4<f32>>,
    
    // Moments du second ordre (variance)
    v: HashMap<String, Array4<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            timestep: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param_id: &str, param: &mut Array4<f32>, grad: &Array4<f32>) {
        self.timestep += 1;
        
        // Initialiser m et v si nécessaire
        let m = self.m
            .entry(param_id.to_string())
            .or_insert_with(|| Array4::zeros(param.dim()));
        
        let v = self.v
            .entry(param_id.to_string())
            .or_insert_with(|| Array4::zeros(param.dim()));
        
        // Mise à jour des moments
        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        
        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        let grad_squared = grad.mapv(|x| x * x);
        *v = &*v * self.beta2 + &grad_squared * (1.0 - self.beta2);
        
        // Correction du biais
        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.timestep as i32)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.timestep as i32)));
        
        // Mise à jour des paramètres
        // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        let update = m_hat.mapv(|m_val| m_val * self.learning_rate);
        let denom = v_hat.mapv(|v_val| v_val.sqrt() + self.epsilon);
        let step = update / denom;
        
        *param -= &step;
    }
    
    fn zero_grad(&mut self) {
        // Adam garde les moments entre les étapes
    }
    
    fn get_lr(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Scheduler de learning rate
pub enum LRScheduler {
    StepDecay {
        optimizer: Box<dyn Optimizer>,
        step_size: usize,
        gamma: f32,
        last_epoch: usize,
    },
    ReduceOnPlateau {
        optimizer: Box<dyn Optimizer>,
        factor: f32,
        patience: usize,
        best_loss: f32,
        num_bad_epochs: usize,
    },
}

impl LRScheduler {
    pub fn step_decay(optimizer: Box<dyn Optimizer>, step_size: usize, gamma: f32) -> Self {
        LRScheduler::StepDecay {
            optimizer,
            step_size,
            gamma,
            last_epoch: 0,
        }
    }
    
    pub fn reduce_on_plateau(optimizer: Box<dyn Optimizer>, factor: f32, patience: usize) -> Self {
        LRScheduler::ReduceOnPlateau {
            optimizer,
            factor,
            patience,
            best_loss: f32::INFINITY,
            num_bad_epochs: 0,
        }
    }
    
    pub fn step_epoch(&mut self) {
        match self {
            LRScheduler::StepDecay { optimizer, step_size, gamma, last_epoch } => {
                *last_epoch += 1;
                if *last_epoch % *step_size == 0 {
                    let new_lr = optimizer.get_lr() * *gamma;
                    optimizer.set_lr(new_lr);
                    println!("📉 Learning rate réduit à: {:.6}", new_lr);
                }
            }
            _ => {}
        }
    }
    
    pub fn step_loss(&mut self, loss: f32) {
        match self {
            LRScheduler::ReduceOnPlateau { 
                optimizer, 
                factor, 
                patience, 
                best_loss, 
                num_bad_epochs 
            } => {
                if loss < *best_loss {
                    *best_loss = loss;
                    *num_bad_epochs = 0;
                } else {
                    *num_bad_epochs += 1;
                    if *num_bad_epochs >= *patience {
                        let new_lr = optimizer.get_lr() * *factor;
                        optimizer.set_lr(new_lr);
                        println!("📉 Plateau détecté - Learning rate réduit à: {:.6}", new_lr);
                        *num_bad_epochs = 0;
                    }
                }
            }
            _ => {}
        }
    }
    
    pub fn get_optimizer_mut(&mut self) -> &mut Box<dyn Optimizer> {
        match self {
            LRScheduler::StepDecay { optimizer, .. } => optimizer,
            LRScheduler::ReduceOnPlateau { optimizer, .. } => optimizer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    
    #[test]
    fn test_sgd_momentum() {
        let mut optimizer = SGD::new(0.01, 0.9);
        let mut param = Array4::ones((2, 2, 2, 2));
        let grad = Array4::ones((2, 2, 2, 2));
        
        let initial_val = param[[0, 0, 0, 0]];
        optimizer.step("test_param", &mut param, &grad);
        
        assert!(param[[0, 0, 0, 0]] < initial_val);
    }
    
    #[test]
    fn test_adam() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut param = Array4::ones((2, 2, 2, 2));
        let grad = Array4::ones((2, 2, 2, 2));
        
        let initial_val = param[[0, 0, 0, 0]];
        optimizer.step("test_param", &mut param, &grad);
        
        assert!(param[[0, 0, 0, 0]] < initial_val);
    }
}
