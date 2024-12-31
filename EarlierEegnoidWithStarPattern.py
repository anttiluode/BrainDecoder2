import os
import time
import logging
import hashlib
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import mne
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

# Additional scientific imports for frequency analysis and advanced computations
from scipy import signal
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is contiguous and has positive strides."""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.copy()

def check_tensor_device(tensor: torch.Tensor, expected_device: torch.device):
    """Check if the tensor is on the expected device."""
    if tensor.device != expected_device:
        raise RuntimeError(f"Tensor on wrong device: {tensor.device} vs {expected_device}")

def log_gpu_memory():
    """Log the current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"GPU Memory Allocated: {allocated:.2f} GB")
        logging.info(f"GPU Memory Reserved: {reserved:.2f} GB")

# -----------------------------------------------------------------------------
# Brain Constraints
# -----------------------------------------------------------------------------
class BrainConstraints:
    """Applies anatomical constraints to neural fields to simulate brain-like structures."""
    def __init__(self, resolution: int = 512, device='cpu'):
        self.resolution = resolution
        self.device = device
        self.cortical_thickness = 2.5  # mm, example value
        
        # Initialize masks on the specified device
        self.white_matter_mask = self._create_white_matter_mask().to(self.device)
        self.gray_matter_mask = self._create_gray_matter_mask().to(self.device)
    
    def _create_white_matter_mask(self) -> torch.Tensor:
        """Creates a mask representing white matter regions."""
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        center = self.resolution // 2
        radius = int(self.resolution * 0.4)
        cv2.circle(mask, (center, center), radius, 1.0, -1)
        return torch.from_numpy(mask).to(torch.float32)
    
    def _create_gray_matter_mask(self) -> torch.Tensor:
        """Creates a mask representing gray matter regions."""
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        center = self.resolution // 2
        radius = int(self.resolution * 0.2)
        cv2.circle(mask, (center, center), radius, 1.0, -1)
        return torch.from_numpy(mask).to(torch.float32)
    
    def apply_constraints(self, field: torch.Tensor) -> torch.Tensor:
        """Applies anatomical constraints to the field."""
        try:
            # Ensure masks are on the same device as the field
            if self.gray_matter_mask.device != field.device:
                self.gray_matter_mask = self.gray_matter_mask.to(field.device)
                self.white_matter_mask = self.white_matter_mask.to(field.device)
            
            # Apply constraints
            constrained = field * self.gray_matter_mask
            
            # Move to CPU for gaussian_filter if needed
            if field.device.type == 'cuda':
                constrained_cpu = constrained.cpu().numpy()
            else:
                constrained_cpu = constrained.numpy()
                
            smoothed = gaussian_filter(constrained_cpu, sigma=self.cortical_thickness)
            
            # Move back to original device
            return torch.from_numpy(smoothed).to(field.device)
            
        except Exception as e:
            logging.error(f"Error applying brain constraints: {str(e)}")
            return field  # Return original field if constraints fail

# -----------------------------------------------------------------------------
# Field Processor (Enhanced with CUDA and Brain Constraints)
# -----------------------------------------------------------------------------
class FieldProcessor:
    """
    Processes EEG information using neural fields based on wave equations.
    Maps lower frequencies (theta/alpha) to coarse structure,
    and higher frequencies (beta/gamma) to fine details.
    """
    def __init__(self, resolution=512, field_dir='fields', dt=0.1,
                spatial_coupling=1.0, temporal_coupling=0.5, device='cpu'):
        self.resolution = resolution
        self.field_dir = field_dir
        os.makedirs(self.field_dir, exist_ok=True)
        self.device = device
        
        # Initialize Brain Constraints with the same device
        self.brain_constraints = BrainConstraints(
            resolution=self.resolution,
            device=self.device
        )
        
        # Define frequency bands and their properties
        self.frequency_bands = {
            'theta': {  # 4-8 Hz - coarse structure
                'range': (4, 8),
                'scale_range': (0.5, 1.0),
                'transforms': 3,
                'detail_weight': 0.2,
                'original_scale_range': (0.5, 1.0)
            },
            'alpha': {  # 8-13 Hz - intermediate structure
                'range': (8, 13),
                'scale_range': (0.3, 0.6),
                'transforms': 4,
                'detail_weight': 0.4,
                'original_scale_range': (0.3, 0.6)
            },
            'beta': {  # 13-30 Hz - fine structure
                'range': (13, 30),
                'scale_range': (0.2, 0.4),
                'transforms': 5,
                'detail_weight': 0.6,
                'original_scale_range': (0.2, 0.4)
            },
            'gamma': {  # 30-100 Hz - finest details
                'range': (30, 100),
                'scale_range': (0.1, 0.3),
                'transforms': 6,
                'detail_weight': 0.8,
                'original_scale_range': (0.1, 0.3)
            }
        }
        
        # Field parameters
        self.field_params = {
            'dt': dt,
            'spatial_coupling': spatial_coupling,
            'temporal_coupling': temporal_coupling
        }
        
        # Field constants for the wave equation
        self.c = 1.0  # Wave speed
        self.alpha = 0.1  # Nonlinear damping
        self.beta = 0.05   # Coupling strength
        
        # Initialize Brain Constraints
        self.brain_constraints = BrainConstraints(resolution=self.resolution)
    
    def generate_deterministic_seed(self, image_path: str) -> int:
        """Generate a deterministic seed based on the image path."""
        hash_digest = hashlib.md5(image_path.encode()).hexdigest()
        return int(hash_digest, 16) % (2**32)
    
    def analyze_eeg_frequencies(self, eeg_data: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract power in different frequency bands from 1D EEG data.

        :param eeg_data: 1D numpy array of the EEG signal (e.g., 1s window).
        :param fs: Sampling frequency of the EEG data.
        :return: dict of band_name -> average power
        """
        band_powers = {}
        # For each band, perform bandpass filtering and compute power
        for band_name, band_info in self.frequency_bands.items():
            low, high = band_info['range']
            nyq = fs / 2
            try:
                b, a = signal.butter(4, [low/nyq, high/nyq], btype='band', analog=False)
                # Filter signal
                filtered = signal.filtfilt(b, a, eeg_data)
                # Power calculation
                band_powers[band_name] = np.mean(filtered**2)
            except Exception as e:
                logging.error(f"Error in frequency analysis for band {band_name}: {e}")
                band_powers[band_name] = 0.0  # Assign zero power if error occurs
                
        return band_powers

    def generate_field_pattern(self, band_powers: Dict[str, float], seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a combined neural field pattern where each band influences the field dynamics.
        
        :param band_powers: dict of band_name -> band power
        :param seed: Optional seed for deterministic field generation
        :return: field pattern tensor (float32, range ~ [0..1]) on GPU
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        field = torch.zeros((self.resolution, self.resolution), device=self.device, dtype=torch.float32)
        
        total_power = sum(band_powers.values())
        if total_power < 1e-12:
            # If no power, or all-zero EEG, just return empty field
            return field
        
        for band_name, power in band_powers.items():
            band_info = self.frequency_bands[band_name]
            # Relative power
            rel_power = power / total_power
            
            # 1) Compute field for this band
            band_field = self._compute_band_field(band_info, rel_power)
            
            # 2) Process field to add dynamics
            processed = self._process_band_field(band_field, band_info, rel_power)
            
            # 3) Weighted sum
            field += processed * band_info['detail_weight']
        
        # Apply brain constraints
        field = self.brain_constraints.apply_constraints(field)
        
        # Normalize final result
        min_val = field.min()
        max_val = field.max()
        if (max_val - min_val) > 1e-12:
            field = (field - min_val) / (max_val - min_val)
        return field
    
    def _compute_band_field(self, band_info: dict, relative_power: float) -> torch.Tensor:
        """
        Compute neural field for a single frequency band using the wave equation.
        
        :param band_info: dictionary containing band properties
        :param relative_power: relative power of the band
        :return: computed field tensor
        """
        # Initialize field u(x, y, t) and velocity v(x, y, t)
        u = torch.rand((self.resolution, self.resolution), device=self.device, dtype=torch.float32) * 0.1
        v = torch.zeros_like(u)
        
        # Time steps for simulation
        num_steps = 100  # Adjust as needed for dynamics
        
        # Grid spacing
        dx = 1.0
        dy = 1.0
        
        for step in range(num_steps):
            # Compute Laplacian using finite differences
            laplacian = (
                torch.roll(u, shifts=1, dims=0) +
                torch.roll(u, shifts=-1, dims=0) +
                torch.roll(u, shifts=1, dims=1) +
                torch.roll(u, shifts=-1, dims=1) -
                4 * u
            ) / (dx * dy)
            
            # Enhanced wave equation with phase coupling
            # ∂²u/∂t² = c²∇²u - αu³ - βv + ξ + Phase Coupling
            # Update velocity v
            quantum_noise = torch.randn_like(v, device=self.device) * 0.05
            # Compute phase using FFT
            fft_u = torch.fft.fft2(u)
            phase = torch.angle(fft_u)
            coupling = torch.real(torch.fft.ifft2(torch.exp(1j * phase))) * self.field_params['spatial_coupling']
            coupling = coupling.to(self.device)
            
            v_update = self.field_params['dt'] * (
                self.c**2 * laplacian - 
                self.alpha * (u ** 3) - 
                self.beta * v + 
                quantum_noise +
                coupling
            )
            v += v_update
            
            # Update field u
            u += self.field_params['dt'] * v
        
        # Apply relative power scaling
        u *= relative_power
        
        # Normalize
        u = (u - u.min()) / (u.max() - u.min() + 1e-12)
        return u

    def _process_band_field(self, field: torch.Tensor, band_info: dict, relative_power: float) -> torch.Tensor:
        """
        Apply multi-scale processing to add details to the field.
        """
        # Create a Gaussian pyramid to extract details
        levels = 3
        pyramid = []
        current = field.cpu().numpy()

        for _ in range(levels):
            h, w = current.shape
            if min(h, w) < 2:
                break
            scaled = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            pyramid.append(scaled)
            current = scaled

        # Reconstruct with frequency-dependent weighting
        processed = np.zeros_like(field.cpu().numpy())
        size = (self.resolution, self.resolution)
        for i, level_img in enumerate(pyramid):
            # Higher-level pyramid = finer detail
            weight = relative_power * (1.0 - i / len(pyramid))
            up = cv2.resize(level_img, size, interpolation=cv2.INTER_LINEAR)
            processed += up * weight

        # Combine with the base field
        processed += field.cpu().numpy() * 0.5

        # Final normalize
        mx = processed.max()
        if mx > 1e-12:
            processed /= mx

        return torch.from_numpy(processed).to(self.device)
    
    def generate_and_save_field(self, eeg_data: np.ndarray, fs: float, seed: Optional[int] = None) -> str:
        """
        Generate neural field from EEG data and save it.
        :param eeg_data: 1D numpy array of EEG data.
        :param fs: Sampling frequency.
        :param seed: Optional seed for deterministic field generation.
        :return: Path to the saved field image.
        """
        band_powers = self.analyze_eeg_frequencies(eeg_data, fs)
        field_tensor = self.generate_field_pattern(band_powers, seed)
        field_np = field_tensor.cpu().numpy()
        field_uint8 = (field_np * 255).astype(np.uint8)
        field_pil = Image.fromarray(field_uint8)
        # Generate a unique filename based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        field_filename = f"field_generated_{timestamp}.png"
        field_path = os.path.join(self.field_dir, field_filename)
        field_pil.save(field_path)
        logging.info(f"Generated and saved field: {field_path}")
        return field_path
    
    def save_field_from_original(self, original_image_path: str, band_powers: Dict[str, float]) -> str:
        """
        Generate and save the field image corresponding to the original image.
        :param original_image_path: Path to the original image.
        :param band_powers: Band powers used for field generation.
        :return: Path to the saved field image.
        """
        field_tensor = self.generate_field_pattern(band_powers)
        field_np = field_tensor.cpu().numpy()
        field_uint8 = (field_np * 255).astype(np.uint8)
        field_pil = Image.fromarray(field_uint8)
        original_basename = os.path.basename(original_image_path)
        if original_basename.startswith('field_'):
            field_filename = original_basename  # Avoid double prefixing
        else:
            field_filename = f"field_{original_basename}"
        field_path = os.path.join(self.field_dir, field_filename)
        field_pil.save(field_path)
        logging.info(f"Saved field image: {field_path}")
        return field_path

# -----------------------------------------------------------------------------
# EEG Processing
# -----------------------------------------------------------------------------
class EEGProcessor:
    """Handles EEG data loading and retrieval."""
    def __init__(self, resolution: int = 512):
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        # 1 second window for retrieval
        self.window_size = 1.0  
        self.resolution = resolution

    def load_file(self, filepath: str) -> bool:
        """Load EEG data from an EDF file."""
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            return True
        except Exception as e:
            logging.error(f"Failed to load EEG file: {e}")
            return False

    def get_channels(self):
        """Return list of channel names."""
        if self.raw:
            return self.raw.ch_names
        return []

    def get_data(self, channel: int, start_time: float) -> Optional[np.ndarray]:
        """Retrieve 1s of EEG data for a specific channel and time."""
        if self.raw is None:
            return None
        try:
            start_sample = int(start_time * self.sfreq)
            samples_needed = int(self.window_size * self.sfreq)
            end_sample = start_sample + samples_needed
            if end_sample > self.raw.n_times:
                end_sample = self.raw.n_times
            data, _ = self.raw[channel, start_sample:end_sample]
            return data.flatten()
        except Exception as e:
            logging.error(f"Error getting EEG data: {e}")
            return None

# -----------------------------------------------------------------------------
# Enhanced U-Net Model (With Gradient Checkpointing and Mixed Precision Compatibility)
# -----------------------------------------------------------------------------
class EnhancedUNet(nn.Module):
    """U-Net for reversing field transformations (outputs grayscale)."""
    def __init__(self):
        super(EnhancedUNet, self).__init__()

        # Encoder
        self.enc1 = self._double_conv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)

        # Final conv => 1 channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """Helper for double convolution layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final_conv(d1)

# -----------------------------------------------------------------------------
# Dataset for Training
# -----------------------------------------------------------------------------
class ImagePairDataset(Dataset):
    """Dataset for image pairs (field and original)."""
    def __init__(self, original_dir, field_dir, image_pairs, transform=None):
        self.original_dir = original_dir
        self.field_dir = field_dir
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        field_file, orig_file = self.image_pairs[idx]  # Order: field first, then original

        # Load images
        field_img = Image.open(os.path.join(self.field_dir, field_file)).convert('L')
        original_img = Image.open(os.path.join(self.original_dir, orig_file)).convert('L')

        if self.transform:
            field_img = self.transform(field_img)
            original_img = self.transform(original_img)

        return field_img, original_img

# -----------------------------------------------------------------------------
# Biological Neural Network Classes
# -----------------------------------------------------------------------------
class BiologicalNeuron:
    def __init__(self, position):
        # Biological properties
        self.position = position
        self.membrane_potential = -70.0  # resting potential in mV
        self.threshold = -55.0  # firing threshold
        self.refractory_period = 2.0  # ms
        self.last_spike_time = 0.0
        self.spike_history = []
        
        # Network properties
        self.connections = []
        self.weights = []
        
    def update(self, t, input_current):
        # Check refractory period
        if t - self.last_spike_time < self.refractory_period:
            return False
        
        # Update membrane potential
        self.membrane_potential += (-70.0 - self.membrane_potential) * 0.1 + input_current
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = -70.0
            self.last_spike_time = t
            self.spike_history.append(t)
            return True
        return False

class BiologicalLayer:
    def __init__(self, size, radius=20, density=1.0):
        self.size = size
        self.radius = radius
        self.density = density
        self.neurons = []
        self.initialize_neurons()
        
    def initialize_neurons(self):
        # Create circular arrangement of neurons
        center = self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                # Check if within circle
                if (i - center)**2 + (j - center)**2 <= self.radius**2:
                    if np.random.random() < self.density:
                        self.neurons.append(BiologicalNeuron((i, j)))
        
        # Create connections
        self.connect_neurons()
        
    def connect_neurons(self):
        for n1 in self.neurons:
            for n2 in self.neurons:
                if n1 != n2:
                    dist = np.sqrt((n1.position[0] - n2.position[0])**2 + 
                                  (n1.position[1] - n2.position[1])**2)
                    if dist < 5:  # Connection radius
                        n1.connections.append(n2)
                        n1.weights.append(np.exp(-dist/5))  # Placeholder weight

class BiologicalNetwork:
    def __init__(self, num_layers=3, size=50, density=1.0):
        self.layers = []
        for i in range(num_layers):
            radius = 20 * (0.8 ** i)  # Each layer gets smaller
            self.layers.append(BiologicalLayer(size, radius, density))
        
        self.time = 0
        self.size = size
        self.activity_history = []
        
    def load_weights_from_eeg(self, eeg_weights):
        """
        Load weights from EEG data for each layer.
        eeg_weights: 2D list or array where eeg_weights[layer][connection] corresponds
                     to the weight for each connection in the layer.
        """
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx >= len(eeg_weights):
                break
            layer_eeg_weights = eeg_weights[layer_idx]
            for neuron in layer.neurons:
                for conn_idx in range(len(neuron.connections)):
                    if conn_idx < len(layer_eeg_weights):
                        neuron.weights[conn_idx] = layer_eeg_weights[conn_idx]
                    else:
                        neuron.weights[conn_idx] = 0.1  # Default small weight
    
    def update(self):
        self.time += 1
        layer_activities = []
        
        for layer_idx, layer in enumerate(self.layers):
            # Update each neuron in the layer
            spikes = []
            activities = np.zeros((self.size, self.size))
            
            for neuron in layer.neurons:
                # Calculate input from connected neurons
                input_current = 0.0
                for conn, weight in zip(neuron.connections, neuron.weights):
                    input_current += weight * (conn.membrane_potential + 70) / 70
                
                # Add input from previous layer if exists
                if layer_idx > 0 and len(self.activity_history) > 0:
                    prev_activities = self.activity_history[-1][layer_idx-1]['activities']
                    i, j = neuron.position
                    # Sum activities in the neighborhood
                    input_current += prev_activities[max(0, i-1):min(self.size, i+2),
                                                     max(0, j-1):min(self.size, j+2)].sum() * 0.1
                
                # Update neuron
                spiked = neuron.update(self.time, input_current)
                if spiked:
                    spikes.append(neuron.position)
                
                # Record activity
                activities[neuron.position] = (neuron.membrane_potential + 70) / 70
            
            layer_activities.append({
                'activities': activities,
                'spikes': spikes
            })
        
        self.activity_history.append(layer_activities)
        if len(self.activity_history) > 100:  # Keep last 100 timesteps
            self.activity_history.pop(0)


@dataclass
class VisualizationState:
    """Holds the current state of visualization."""
    current_time: float = 0.0
    is_playing: bool = False
    seek_requested: bool = False
    seek_time: float = 0.0
    playback_speed: float = 1.0

# -----------------------------------------------------------------------------
# Biological Bridge
# -----------------------------------------------------------------------------
class BiologicalBridge:
    def __init__(self, input_size=50, num_layers=3, device='cpu'):
        self.input_size = input_size
        self.device = device
        self.plasticity_rate = 0.01
        self.adaptation_threshold = 0.5
        self.resonance_window = 20  # ms
        
        # Initialize biological network
        self.bio_network = BiologicalNetwork(num_layers=num_layers, size=input_size)
        
        # Resonance detection
        self.resonance_buffer = []
        self.resonance_patterns = {}
        
        # Plasticity and adaptation parameters
        self.synaptic_tags = {}  # For marking active synapses
        self.meta_plasticity = {}  # For tracking long-term changes
        
        # Initialize frequency bands parameters
        self.freq_bands = {
            'theta': {'range': (4, 8), 'scale': 2.0, 'weight': 0.3},
            'alpha': {'range': (8, 13), 'scale': 1.0, 'weight': 0.3},
            'beta': {'range': (13, 30), 'scale': 0.5, 'weight': 0.2},
            'gamma': {'range': (30, 100), 'scale': 0.25, 'weight': 0.2}
        }
        
    def process_eeg_pattern(self, eeg_data, nn_activations):
        """
        Process EEG data while considering neural network activations
        Returns both biological response and potential resonance patterns
        """
        try:
            # Ensure data is on correct device
            eeg_data = torch.tensor(eeg_data, device=self.device) if isinstance(eeg_data, np.ndarray) else eeg_data
            nn_activations = torch.tensor(nn_activations, device=self.device) if isinstance(nn_activations, np.ndarray) else nn_activations
            
            # Convert EEG to biological activity patterns
            bio_pattern = self._eeg_to_bio_pattern(eeg_data)
            
            # Process through biological network
            bio_response = self.bio_network.update_with_feedback(bio_pattern, nn_activations)
            
            # Detect resonance between bio and nn patterns
            resonance = self._detect_resonance(bio_response, nn_activations)
            
            # Update plasticity based on resonance
            if resonance > self.adaptation_threshold:
                self._update_plasticity(bio_response, nn_activations)
            
            emergent_patterns = self._extract_emergent_patterns()
            
            return {
                'bio_response': bio_response,
                'resonance': resonance,
                'emergent_patterns': emergent_patterns
            }
        except Exception as e:
            logging.error(f"Error in biological bridge processing: {str(e)}")
            return None
    
    def _eeg_to_bio_pattern(self, eeg_data):
        """Transform EEG data into biological activity pattern"""
        spatial_pattern = torch.zeros((self.input_size, self.input_size), device=self.device)
        
        for band_name, band_info in self.freq_bands.items():
            # Extract band-specific activity
            filtered = self._bandpass_filter(eeg_data, *band_info['range'])
            
            # Create spatial pattern for this band
            band_pattern = self._create_spatial_waves(filtered, band_info['scale'])
            
            if band_name == 'alpha':
                # Alpha modulates inhibition
                spatial_pattern *= (1.0 + band_pattern * band_info['weight'])
            else:
                # Other bands add their patterns
                spatial_pattern += band_pattern * band_info['weight']
        
        return spatial_pattern
    
    def _detect_resonance(self, bio_response, nn_pattern):
        """Detect resonant patterns between biological and NN activities"""
        try:
            # Add current patterns to resonance buffer
            self.resonance_buffer.append((bio_response, nn_pattern))
            if len(self.resonance_buffer) > self.resonance_window:
                self.resonance_buffer.pop(0)
            
            # Calculate temporal coherence across the window
            temporal_coherence = self._compute_temporal_coherence()
            
            # Calculate spatial similarity
            spatial_similarity = self._compute_spatial_similarity(bio_response, nn_pattern)
            
            # Calculate frequency coherence
            freq_coherence = self._compute_frequency_coherence(bio_response, nn_pattern)
            
            # Combine measures with weights
            resonance_score = (0.4 * temporal_coherence + 
                             0.3 * spatial_similarity +
                             0.3 * freq_coherence)
            
            return float(resonance_score)
        except Exception as e:
            logging.error(f"Error in resonance detection: {str(e)}")
            return 0.0
    
    def _bandpass_filter(self, data, low, high):
        """Apply a bandpass filter to the data using FFT"""
        try:
            # Get frequency components
            fft = torch.fft.rfft(data)
            freqs = torch.fft.rfftfreq(len(data), 1/256.0)  # Assuming 256 Hz sampling
            
            # Create bandpass mask
            mask = (freqs >= low) & (freqs <= high)
            mask = mask.to(self.device)
            
            # Apply filter
            fft_filtered = fft * mask
            
            # Inverse FFT
            filtered = torch.fft.irfft(fft_filtered)
            return filtered
            
        except Exception as e:
            logging.error(f"Error in bandpass filter: {str(e)}")
            return torch.zeros_like(data)
    
    def _create_spatial_waves(self, signal_data, scale=1.0):
        """Create spatial wave patterns based on the signal data"""
        try:
            # Create coordinate grid
            x = torch.linspace(-1, 1, self.input_size, device=self.device)
            y = torch.linspace(-1, 1, self.input_size, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            R = torch.sqrt(X*X + Y*Y)
            
            # Create modulated pattern
            pattern = torch.zeros_like(R)
            signal = signal_data.to(self.device)
            
            # Modulate Gaussian envelope with signal
            for i, amp in enumerate(signal):
                pattern += amp * torch.exp(-R**2/(2*scale**2)) * torch.cos(2*np.pi*i/len(signal))
            
            return pattern
            
        except Exception as e:
            logging.error(f"Error creating spatial waves: {str(e)}")
            return torch.zeros((self.input_size, self.input_size), device=self.device)
    
    def _compute_temporal_coherence(self):
        """Compute temporal coherence over the resonance window"""
        if len(self.resonance_buffer) < 2:
            return 0.0
            
        try:
            # Extract temporal sequences
            bio_sequence = torch.stack([b[0] for b in self.resonance_buffer])
            nn_sequence = torch.stack([b[1] for b in self.resonance_buffer])
            
            # Compute correlation over time
            correlation = torch.corrcoef(bio_sequence.flatten(), nn_sequence.flatten())[0, 1]
            return abs(float(correlation))
        except Exception as e:
            logging.error(f"Error computing temporal coherence: {str(e)}")
            return 0.0
    
    def _compute_spatial_similarity(self, bio_pattern, nn_pattern):
        """Compute spatial similarity between biological and NN patterns"""
        try:
            bio_flat = bio_pattern.flatten()
            nn_flat = nn_pattern.flatten()
            
            # Normalize patterns
            bio_norm = torch.norm(bio_flat)
            nn_norm = torch.norm(nn_flat)
            
            if bio_norm == 0 or nn_norm == 0:
                return 0.0
                
            bio_normalized = bio_flat / bio_norm
            nn_normalized = nn_flat / nn_norm
            
            # Compute cosine similarity
            similarity = torch.dot(bio_normalized, nn_normalized)
            return float(similarity)
        except Exception as e:
            logging.error(f"Error computing spatial similarity: {str(e)}")
            return 0.0
    
    def _compute_frequency_coherence(self, bio_pattern, nn_pattern):
        """Compute coherence in frequency domain"""
        try:
            # Compute FFT
            bio_fft = torch.fft.rfft2(bio_pattern)
            nn_fft = torch.fft.rfft2(nn_pattern)
            
            # Get power spectra
            bio_power = torch.abs(bio_fft)
            nn_power = torch.abs(nn_fft)
            
            # Normalize
            bio_power_norm = bio_power / torch.sum(bio_power)
            nn_power_norm = nn_power / torch.sum(nn_power)
            
            # Compute correlation between power spectra
            coherence = torch.corrcoef(
                bio_power_norm.flatten(),
                nn_power_norm.flatten()
            )[0, 1]
            
            return float(abs(coherence))
        except Exception as e:
            logging.error(f"Error computing frequency coherence: {str(e)}")
            return 0.0
    
    def _update_plasticity(self, bio_pattern, nn_pattern):
        """Update synaptic weights based on detected resonance"""
        try:
            # Find active regions in both patterns
            bio_active = (bio_pattern > bio_pattern.mean() + bio_pattern.std()).float()
            nn_active = (nn_pattern > nn_pattern.mean() + nn_pattern.std()).float()
            
            # Find coincidentally active regions
            coincident = bio_active * nn_active
            
            # Update synaptic tags for coincident regions
            for i in range(self.input_size):
                for j in range(self.input_size):
                    if coincident[i, j] > 0:
                        synapse_id = f"synapse_{i}_{j}"
                        
                        # Initialize or update synaptic tag
                        if synapse_id not in self.synaptic_tags:
                            self.synaptic_tags[synapse_id] = 0
                        self.synaptic_tags[synapse_id] += 1
                        
                        # Apply meta-plasticity for frequently active synapses
                        if self.synaptic_tags[synapse_id] > 10:
                            if synapse_id not in self.meta_plasticity:
                                self.meta_plasticity[synapse_id] = 1.0
                            self.meta_plasticity[synapse_id] *= 1.1
            
            self._apply_weight_updates()
            
        except Exception as e:
            logging.error(f"Error updating plasticity: {str(e)}")
    
    def _extract_emergent_patterns(self):
        """Extract emergent patterns from the biological network"""
        try:
            patterns = []
            
            # Get current network state
            current_state = self.bio_network.get_state()
            
            for layer_idx, layer_state in enumerate(current_state):
                # Find active assemblies
                assemblies = self._find_neural_assemblies(layer_state)
                
                for assembly in assemblies:
                    # Get temporal evolution
                    temporal_pattern = self._get_assembly_temporal_pattern(assembly)
                    
                    # Classify pattern type
                    pattern_type = self._classify_pattern(temporal_pattern)
                    
                    patterns.append({
                        'layer': layer_idx,
                        'assembly': assembly,
                        'pattern_type': pattern_type,
                        'temporal_signature': temporal_pattern
                    })
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error extracting emergent patterns: {str(e)}")
            return []
    
    def _find_neural_assemblies(self, layer_state):
        """Find neural assemblies using clustering"""
        try:
            # Threshold for active neurons
            threshold = layer_state.mean() + layer_state.std()
            
            # Find active regions
            active = (layer_state > threshold).float()
            
            # Use connected components to find assemblies
            from scipy import ndimage
            labeled, num_features = ndimage.label(active.cpu().numpy())
            
            assemblies = []
            for i in range(1, num_features + 1):
                points = np.argwhere(labeled == i)
                if len(points) >= 3:  # Minimum size for an assembly
                    assemblies.append(points)
            
            return assemblies
            
        except Exception as e:
            logging.error(f"Error finding neural assemblies: {str(e)}")
            return []
    
    def _get_assembly_temporal_pattern(self, assembly):
        """Extract temporal pattern for an assembly"""
        try:
            # Get activity history for assembly region
            history = []
            
            for state in self.resonance_buffer:
                region_activity = torch.tensor(
                    [state[0][tuple(point)] for point in assembly]
                ).mean()
                history.append(float(region_activity))
            
            return torch.tensor(history, device=self.device)
            
        except Exception as e:
            logging.error(f"Error getting assembly temporal pattern: {str(e)}")
            return torch.zeros(self.resonance_window, device=self.device)
    
    def _classify_pattern(self, temporal_pattern):
        """Classify temporal pattern type"""
        try:
            # Compute FFT of temporal pattern
            fft = torch.fft.rfft(temporal_pattern)
            power = torch.abs(fft)
            
            # Normalize power spectrum
            power_norm = power / power.sum()
            
            # Get dominant frequency
            max_freq_idx = torch.argmax(power_norm)
            
            # Classify based on spectrum properties
            if max_freq_idx > len(power_norm) * 0.6:
                return 'oscillatory'  # High-frequency oscillation
            elif power_norm[0] > 0.6:
                return 'sustained'    # Strong DC component
            else:
                return 'burst'        # Mixed frequency content
                
        except Exception as e:
            logging.error(f"Error classifying pattern: {str(e)}")
            return 'unknown'

# -----------------------------------------------------------------------------
# Decoder Testing
# -----------------------------------------------------------------------------
class DecoderTest:
    """Testing framework for field-to-image decoding."""
    def __init__(self, model, device, resolution=512, field_dir='fields'):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.field_dir = field_dir  # Directory where fields are stored
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        # We'll create one field processor for testing
        self.field_processor = FieldProcessor(resolution=self.resolution, field_dir=self.field_dir, device=self.device.type)

    def process_paired_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Test field encoding and decoding on a paired image (original and field).
        Returns: (original, field, decoded, PSNR, SSIM)
        """
        # Load and preprocess original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Invalid image or cannot open: {image_path}")

        # Resize to match the resolution
        original = cv2.resize(original, (self.resolution, self.resolution))

        # Analyze EEG frequencies (simulated here as uniform band powers for demonstration)
        # In actual use, band_powers should come from EEG data
        band_powers = {
            'theta': 1.0,
            'alpha': 0.5,
            'beta': 1.5,
            'gamma': 2.0
        }

        # Generate and save field
        field_path = self.field_processor.save_field_from_original(image_path, band_powers)

        # Load the corresponding field image
        # Assuming field images are named as 'field_<original_filename>'
        original_basename = os.path.basename(image_path)
        if original_basename.startswith('field_'):
            field_filename = original_basename  # Avoid double prefixing
        else:
            field_filename = f"field_{original_basename}"
        field_path = os.path.join(self.field_dir, field_filename)
        field = cv2.imread(field_path, cv2.IMREAD_GRAYSCALE)
        if field is None:
            raise ValueError(f"Corresponding field image not found: {field_path}")

        # Resize field to match resolution if necessary
        field = cv2.resize(field, (self.resolution, self.resolution))

        # Decode field
        field_tensor = self.transform(Image.fromarray(field)).unsqueeze(0).to(self.device)

        # Ensure tensor is contiguous
        field_tensor = field_tensor.contiguous()

        # Error checking
        check_tensor_device(field_tensor, self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                decoded_tensor = self.model(field_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                if decoded_np.shape[0] > 1:
                    logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
                decoded_np = decoded_np[0]  # channel 0
                decoded_np = (decoded_np * 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        # Metrics
        if original.shape != decoded_np.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape}, decoded={decoded_np.shape}. "
                "They must be identical for PSNR/SSIM."
            )

        psnr_val = psnr(original, decoded_np)
        ssim_val = ssim(original, decoded_np)

        return original, field, decoded_np, psnr_val, ssim_val

    def decode_any_field_image(self, field_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode any standalone field image without needing a paired original.
        Returns: (field, decoded)
        """
        field = cv2.imread(field_path, cv2.IMREAD_GRAYSCALE)
        if field is None:
            raise ValueError(f"Invalid field image or cannot open: {field_path}")

        # Resize field to match resolution if necessary
        field = cv2.resize(field, (self.resolution, self.resolution))

        # Decode field
        field_tensor = self.transform(Image.fromarray(field)).unsqueeze(0).to(self.device)

        # Ensure tensor is contiguous
        field_tensor = field_tensor.contiguous()

        # Error checking
        check_tensor_device(field_tensor, self.device)
        log_gpu_memory()

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                decoded_tensor = self.model(field_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                if decoded_np.shape[0] > 1:
                    logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
                decoded_np = decoded_np[0]  # channel 0
                decoded_np = (decoded_np * 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        return field, decoded_np

# -----------------------------------------------------------------------------
# Video Recording
# -----------------------------------------------------------------------------
class VideoRecorder:
    """Handles recording of EEG visualization to video."""
    def __init__(self, resolution=512):
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.output_path = None

    def start_recording(self, output_path: str, fps: float = 30.0):
        """Start recording video."""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.resolution * 2, self.resolution)
        )
        self.is_recording = True
        logging.info(f"Started recording to {output_path}")

    def add_frame(self, field_frame: np.ndarray, decoded_frame: np.ndarray):
        """Add a frame to the video (side-by-side)."""
        if not self.is_recording:
            return

        field_frame = cv2.resize(field_frame, (self.resolution, self.resolution))
        decoded_frame = cv2.resize(decoded_frame, (self.resolution, self.resolution))
        combined = np.hstack([field_frame, decoded_frame])
        self.writer.write(combined)

    def stop_recording(self):
        """Stop recording and save video."""
        if self.writer:
            self.writer.release()
        self.is_recording = False
        self.writer = None
        logging.info(f"Stopped recording and saved to {self.output_path}")

# -----------------------------------------------------------------------------
# Results Logger
# -----------------------------------------------------------------------------
class ResultsLogger:
    """Logs and tracks decoder performance metrics."""
    def __init__(self, log_dir: str = "decoder_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = []

    def log_result_paired(self, original_path: str, psnr_val: float, ssim_val: float):
        """Log metrics for a paired decode attempt."""
        self.metrics.append({
            'type': 'Paired',
            'image': original_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'timestamp': time.time()
        })
        logging.info(f"Logged paired results for {original_path}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

    def log_result_standalone(self, field_path: str, decoded_image: np.ndarray):
        """Log metrics for a standalone decode attempt."""
        # Since there's no original image, metrics like PSNR and SSIM are not applicable
        self.metrics.append({
            'type': 'Standalone',
            'image': field_path,
            'decoded': decoded_image,
            'timestamp': time.time()
        })
        logging.info(f"Logged standalone decode for {field_path}")

    def save_metrics(self):
        """Save all metrics to file."""
        if not self.metrics:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(self.log_dir, f"metrics_{timestamp}.txt")
        with open(log_path, 'w') as f:
            for m in self.metrics:
                if m['type'] == 'Paired':
                    f.write(f"{m['type']},{m['image']},{m['psnr']:.4f},{m['ssim']:.4f}\n")
                elif m['type'] == 'Standalone':
                    f.write(f"{m['type']},{m['image']},Decoded Image Saved\n")
        logging.info(f"Saved metrics to {log_path}")

    def get_summary(self) -> dict:
        """Get summary statistics of all paired metrics."""
        psnr_vals = [m['psnr'] for m in self.metrics if m['type'] == 'Paired']
        ssim_vals = [m['ssim'] for m in self.metrics if m['type'] == 'Paired']

        if not psnr_vals:
            return {'psnr_avg': 0, 'ssim_avg': 0, 'psnr_std': 0, 'ssim_std': 0}

        return {
            'psnr_avg': np.mean(psnr_vals),
            'psnr_std': np.std(psnr_vals),
            'ssim_avg': np.mean(ssim_vals),
            'ssim_std': np.std(ssim_vals)
        }

# -----------------------------------------------------------------------------
# Enhanced Biological Network with Bridge Integration
# -----------------------------------------------------------------------------
class EnhancedBiologicalNetwork(BiologicalNetwork):
    def __init__(self, num_layers=3, size=50, density=1.0, bridge: Optional[BiologicalBridge] = None):
        super().__init__(num_layers, size, density)
        self.bridge = bridge
        
        # Enhanced biological properties
        self.adaptation_rate = 0.1
        self.homeostatic_target = 0.1
        self.lateral_inhibition = True
        
    def update_with_feedback(self, input_pattern, nn_feedback=None):
        """Update network with both bottom-up and top-down signals"""
        # Update time and network
        super().update()
        
        if nn_feedback is not None:
            # Integrate neural network feedback
            bridge_response = self.bridge.process_eeg_pattern(input_pattern, nn_feedback)
            
            # Apply feedback to each layer
            for layer_idx, layer in enumerate(self.layers):
                self._apply_feedback_modulation(layer, bridge_response, layer_idx)
        
        # Apply homeostatic plasticity
        self._apply_homeostasis()
        
        return self._get_network_state()
    
    def _apply_feedback_modulation(self, layer, bridge_response, layer_idx):
        """Apply top-down modulation based on bridge response"""
        resonance = bridge_response['resonance']
        emergent_patterns = bridge_response['emergent_patterns']
        
        # Modulate neuron properties based on resonance
        for neuron in layer.neurons:
            # Adjust threshold based on resonance
            neuron.threshold += (resonance - 0.5) * self.adaptation_rate
            
            # Apply pattern-specific modulation
            for pattern in emergent_patterns:
                if pattern['layer'] == layer_idx:
                    self._apply_pattern_specific_modulation(neuron, pattern)
    
    def _apply_pattern_specific_modulation(self, neuron, pattern):
        """Apply specific modulation based on detected patterns"""
        if pattern['pattern_type'] == 'oscillatory':
            # Enhance oscillatory behavior
            neuron.membrane_potential += np.sin(self.time * 0.1) * 0.1
        elif pattern['pattern_type'] == 'burst':
            # Enhance burst probability
            neuron.threshold *= 0.95
        elif pattern['pattern_type'] == 'sustained':
            # Enhance sustained activity
            neuron.refractory_period *= 0.9

    def _apply_homeostasis(self):
        """Apply homeostatic plasticity to maintain stability"""
        for layer in self.layers:
            for neuron in layer.neurons:
                if neuron.membrane_potential < self.homeostatic_target:
                    neuron.membrane_potential += 0.5  # Adjust as needed

    def _get_network_state(self):
        """Retrieve the current state of the network for communication"""
        state = []
        for layer in self.layers:
            activities = np.zeros((self.size, self.size))
            for neuron in layer.neurons:
                activities[neuron.position] = (neuron.membrane_potential + 70) / 70
            state.append({'activities': activities, 'spikes': [n.spike_history for n in layer.neurons]})
        return state

# -----------------------------------------------------------------------------
# EEGNoids Application Class
# -----------------------------------------------------------------------------
class EEGNoidsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEGNoids - EEG Neural Organoid Visualizer")
        
        # Initialize components
        self.resolution = 512
        self.eeg = EEGProcessor(resolution=self.resolution)
        self.state = VisualizationState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedUNet().to(self.device)
        self.model.eval()  # default to eval mode
        
        self.last_update = time.time()
        
        # Initialize field processor with a specified field directory
        self.field_dir = 'fields'
        self.field_processor = FieldProcessor(
            resolution=self.resolution,
            field_dir=self.field_dir,
            dt=0.1,
            spatial_coupling=1.0,
            temporal_coupling=0.5,
            device=self.device.type
        )
        
        # Initialize Biological Bridge
        self.biological_bridge = BiologicalBridge(input_size=50, num_layers=3)
        self.enhanced_bio_network = EnhancedBiologicalNetwork(
            num_layers=3, size=50, density=1.0, bridge=self.biological_bridge
        )
        
        # Additional components
        self.decoder_test = DecoderTest(self.model, self.device, self.resolution, field_dir=self.field_dir)
        self.video_recorder = VideoRecorder(self.resolution)
        self.results_logger = ResultsLogger()
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the graphical user interface."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
    
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG", command=self.load_eeg)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
        # Testing menu
        test_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Testing", menu=test_menu)
        test_menu.add_command(label="Test Paired Image", command=self.test_paired_image)
        test_menu.add_command(label="Decode Field Image", command=self.decode_field_image)
        test_menu.add_command(label="Test Batch Images", command=self.test_batch_images)
        test_menu.add_command(label="View Test Results", command=self.view_test_results)
    
        # Processing menu
        process_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=process_menu)
        process_menu.add_command(label="Batch Process Images", command=self.batch_process)
        process_menu.add_command(label="Train Model", command=self.train_model)
    
        # Recording menu
        record_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Recording", menu=record_menu)
        record_menu.add_command(label="Start Recording", command=self.start_recording)
        record_menu.add_command(label="Stop Recording", command=self.stop_recording)
    
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        # Left panel
        control_frame = ttk.LabelFrame(main_container, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
    
        ttk.Label(control_frame, text="EEG Channel:").pack(pady=5)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var)
        self.channel_combo.pack(fill=tk.X, padx=5, pady=5)
    
        play_frame = ttk.Frame(control_frame)
        play_frame.pack(fill=tk.X, pady=5)
        self.play_btn = ttk.Button(play_frame, text="Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)
    
        ttk.Label(control_frame, text="Time Position (s):").pack(pady=5)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(control_frame, from_=0, to=100,
                                     variable=self.time_var, command=self.seek)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)
    
        # Frequency Controls
        self.add_frequency_controls(control_frame)
    
        # Right visualization panel
        viz_frame = ttk.LabelFrame(main_container, text="Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
    
        field_frame = ttk.Frame(viz_frame)
        field_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(field_frame, text="Neural Field Pattern").pack()
        self.field_canvas = tk.Canvas(field_frame, bg='black', width=512, height=512)
        self.field_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
        decoded_frame = ttk.Frame(viz_frame)
        decoded_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(decoded_frame, text="Decoded Image").pack()
        self.decoded_canvas = tk.Canvas(decoded_frame, bg='black', width=512, height=512)
        self.decoded_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # -------------------------------------------------------------------------
    # GUI Command Methods
    # -------------------------------------------------------------------------
    def load_eeg(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if filepath and self.eeg.load_file(filepath):
            self.channel_combo['values'] = self.eeg.get_channels()
            self.channel_combo.set(self.eeg.get_channels()[0])
            self.time_slider.configure(to=self.eeg.duration)
            messagebox.showinfo("Success", "EEG file loaded successfully")
            logging.info(f"EEG file loaded: {filepath}")
            self.update_visualization()
        else:
            messagebox.showerror("Error", "Failed to load EEG file")
            logging.error("Failed to load EEG file")

    def toggle_playback(self):
        self.state.is_playing = not self.state.is_playing
        self.play_btn.configure(text="Pause" if self.state.is_playing else "Play")
        if self.state.is_playing:
            self.last_update = time.time()
            self.update()

    def seek(self, value):
        """When user drags the time slider."""
        if not self.state.is_playing:
            self.state.seek_requested = True
            self.state.seek_time = float(value)
            self.state.current_time = float(value)
            self.update_visualization()

    def update(self):
        """Called repeatedly during playback."""
        if not self.state.is_playing:
            return

        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        self.state.current_time += dt * self.state.playback_speed

        # Loop around if at end
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0

        self.time_var.set(self.state.current_time)
        self.update_visualization()
        self.root.after(33, self.update)  # ~30 FPS

    def update_visualization(self):
        """Compute field from current EEG snippet and decode it."""
        if not self.eeg.raw or not self.channel_var.get():
            return
        try:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
            data = self.eeg.get_data(channel_idx, self.state.current_time)
            if data is None:
                return

            # Generate field pattern
            field_path = self.field_processor.generate_and_save_field(
                data, self.eeg.sfreq,
                seed=self.field_processor.generate_deterministic_seed(self.eeg.raw.filenames[0])
            )

            # Load and process field
            field = cv2.imread(field_path, cv2.IMREAD_GRAYSCALE)
            if field is None:
                logging.error(f"Failed to load field: {field_path}")
                return

            # Resize and prepare field
            field = cv2.resize(field, (self.resolution, self.resolution))
            field_tensor = torch.from_numpy(field).float().to(self.device) / 255.0
            field_tensor = field_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

            # Create visualization
            colored_field = cv2.applyColorMap(field, cv2.COLORMAP_JET)
            field_img = Image.fromarray(cv2.cvtColor(colored_field, cv2.COLOR_BGR2RGB))
            field_img = field_img.resize((512, 512), Image.LANCZOS)
            field_photo = ImageTk.PhotoImage(field_img)
            
            # Update field canvas
            self.field_canvas.delete("all")
            self.field_canvas.create_image(0, 0, image=field_photo, anchor=tk.NW)
            self.field_canvas.image = field_photo  # Keep reference

            # Decode field
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                with torch.no_grad():
                    decoded = self.model(field_tensor)
                    decoded = decoded.squeeze().cpu().numpy()
                    decoded = (decoded * 255).clip(0, 255).astype(np.uint8)
                    decoded = cv2.resize(decoded, (self.resolution, self.resolution))

            # Process through biological bridge
            bio_bridge_output = self.biological_bridge.process_eeg_pattern(
                data,
                decoded.flatten()
            )

            if bio_bridge_output:
                # Use bridge output to modulate decoded image
                bio_response = bio_bridge_output['bio_response']
                resonance = bio_bridge_output['resonance']
                
                # Apply subtle modulation based on resonance
                modulation = np.clip(resonance * 1.2, 0.8, 1.2)
                decoded = (decoded.astype(float) * modulation).clip(0, 255).astype(np.uint8)

            # Update decoded image display
            decoded_img = Image.fromarray(decoded)
            decoded_img = decoded_img.resize((512, 512), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(decoded_img)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = decoded_photo  # Keep reference

            # Handle recording if active
            if self.video_recorder.is_recording:
                decoded_bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
                self.video_recorder.add_frame(colored_field, decoded_bgr)

        except Exception as e:
            logging.error(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_frequency_controls(self, ctrl_frame):
        """Add frequency band control sliders to GUI"""
        freq_frame = ttk.LabelFrame(ctrl_frame, text="Frequency Controls")
        freq_frame.pack(fill=tk.X, padx=5, pady=5)

        self.freq_controls = {}

        # For each frequency band, add power, coupling, and frequency sliders
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            band_frame = ttk.LabelFrame(freq_frame, text=f"{band.title()}")
            band_frame.pack(fill=tk.X, padx=5, pady=2)

            # Power control
            ttk.Label(band_frame, text="Power:").pack(anchor='w')
            power_var = tk.DoubleVar(value=1.0)
            power_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=power_var,
                command=lambda v, b=band: self.update_band_power(b, float(v))
            )
            power_slider.pack(fill=tk.X, padx=5, pady=2)
            power_slider.set(1.0)  # Reset to default

            # Coupling control
            ttk.Label(band_frame, text="Coupling:").pack(anchor='w')
            coupling_var = tk.DoubleVar(value=1.0)
            coupling_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=coupling_var,
                command=lambda v, b=band: self.update_band_coupling(b, float(v))
            )
            coupling_slider.pack(fill=tk.X, padx=5, pady=2)
            coupling_slider.set(1.0)  # Reset to default

            # Oscillation frequency control
            ttk.Label(band_frame, text="Frequency (Hz):").pack(anchor='w')
            freq_var = tk.DoubleVar(value=(
                self.field_processor.frequency_bands[band]['range'][0] + 
                self.field_processor.frequency_bands[band]['range'][1]
            ) / 2)
            freq_slider = ttk.Scale(
                band_frame, 
                from_=self.field_processor.frequency_bands[band]['range'][0],
                to=self.field_processor.frequency_bands[band]['range'][1],
                variable=freq_var,
                command=lambda v, b=band: self.update_band_frequency(b, float(v))
            )
            freq_slider.pack(fill=tk.X, padx=5, pady=2)
            freq_slider.set(freq_var.get())  # Reset to default

            self.freq_controls[band] = {
                'power': power_var,
                'coupling': coupling_var,
                'frequency': freq_var
            }

    def update_band_power(self, band, value):
        """Update band power scaling"""
        # Reset to original scale ranges
        original_scale_min, original_scale_max = self.field_processor.frequency_bands[band]['original_scale_range']
        # Apply scaling based on slider value
        new_scale_min = original_scale_min * value
        new_scale_max = original_scale_max * value
        self.field_processor.frequency_bands[band]['scale_range'] = (new_scale_min, new_scale_max)
        logging.info(f"Updated power scaling for {band} band to {value}")

        # Update visualization
        self.update_visualization()

    def update_band_coupling(self, band, value):
        """Update coupling strength for band"""
        # Here, coupling influences the spatial_coupling parameter
        self.field_processor.field_params['spatial_coupling'] = value
        logging.info(f"Updated spatial coupling for {band} band to {value}")

        # Update visualization
        self.update_visualization()

    def update_band_frequency(self, band, value):
        """Update oscillation frequency"""
        # Update the frequency range based on the center frequency
        band_info = self.field_processor.frequency_bands[band]
        center_freq = value
        bandwidth = (band_info['range'][1] - band_info['range'][0]) / 2
        new_low = max(center_freq - bandwidth / 2, 0.1)  # Prevent negative frequencies
        new_high = center_freq + bandwidth / 2
        if new_high <= new_low:
            new_high = new_low + 0.1  # Ensure high > low

        # Update the frequency range
        band_info['range'] = (new_low, new_high)
        logging.info(f"Updated frequency range for {band} band to {new_low:.2f} - {new_high:.2f} Hz")

        # Update visualization
        self.update_visualization()

    # -------------------------------------------------------------------------
    # Testing / Batch
    # -------------------------------------------------------------------------
    def test_paired_image(self):
        """Test decoding on a paired original-field image."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            results_window = tk.Toplevel(self.root)
            results_window.title("Decoder Test Results (Paired)")

            original, field, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(filepath)

            images_frame = ttk.Frame(results_window)
            images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Original Image
            original_frame = ttk.LabelFrame(images_frame, text="Original Image")
            original_frame.pack(side=tk.LEFT, padx=5)
            original_canvas = tk.Canvas(original_frame, width=256, height=256)
            original_canvas.pack()
            original_pil = Image.fromarray(original)
            original_pil = original_pil.resize((256, 256), Image.LANCZOS)
            original_photo = ImageTk.PhotoImage(original_pil)
            original_canvas.create_image(0, 0, image=original_photo, anchor=tk.NW)
            original_canvas.image = original_photo

            # Field Image
            field_frame = ttk.LabelFrame(images_frame, text="Neural Field")
            field_frame.pack(side=tk.LEFT, padx=5)
            field_canvas = tk.Canvas(field_frame, width=256, height=256)
            field_canvas.pack()
            field_pil = Image.fromarray(field)
            field_pil = field_pil.resize((256, 256), Image.LANCZOS)
            field_photo = ImageTk.PhotoImage(field_pil)
            field_canvas.create_image(0, 0, image=field_photo, anchor=tk.NW)
            field_canvas.image = field_photo

            # Decoded Image
            decoded_frame = ttk.LabelFrame(images_frame, text="Decoded Image")
            decoded_frame.pack(side=tk.LEFT, padx=5)
            decoded_canvas = tk.Canvas(decoded_frame, width=256, height=256)
            decoded_canvas.pack()
            decoded_pil = Image.fromarray(decoded)
            decoded_pil = decoded_pil.resize((256, 256), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(decoded_pil)
            decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            decoded_canvas.image = decoded_photo

            # Metrics
            metrics_frame = ttk.Frame(results_window)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(metrics_frame, text=f"PSNR: {psnr_val:.2f} dB").pack(side=tk.LEFT, padx=10)
            ttk.Label(metrics_frame, text=f"SSIM: {ssim_val:.4f}").pack(side=tk.LEFT, padx=10)

            # Log results
            self.results_logger.log_result_paired(filepath, psnr_val, ssim_val)

        except Exception as e:
            logging.error(f"Error in paired decoder test: {e}")
            messagebox.showerror("Error", f"Paired Test failed: {str(e)}")

    def decode_field_image(self, field_tensor):
        """Decode field tensor into image using the U-Net model"""
        try:
            # Ensure field tensor is on the correct device and has correct shape
            if not isinstance(field_tensor, torch.Tensor):
                field_tensor = torch.from_numpy(field_tensor).float()
            
            # Add batch and channel dimensions if needed
            if len(field_tensor.shape) == 2:
                field_tensor = field_tensor.unsqueeze(0).unsqueeze(0)
                
            # Move to correct device
            field_tensor = field_tensor.to(self.device)
            
            # Normalize if needed
            if field_tensor.max() > 1.0:
                field_tensor = field_tensor / 255.0
                
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                with torch.no_grad():
                    # Run through model
                    decoded = self.model(field_tensor)
                    
                    # Move back to CPU and convert to numpy
                    decoded = decoded.cpu().squeeze().numpy()
                    
                    # Normalize to uint8 range
                    decoded = (decoded * 255).clip(0, 255).astype(np.uint8)
                    
                    # Ensure correct size
                    decoded = cv2.resize(decoded, (self.resolution, self.resolution))
                    
            return decoded
            
        except Exception as e:
            logging.error(f"Decoding error: {str(e)}")
            return np.zeros((self.resolution, self.resolution), dtype=np.uint8)
    
    def test_batch_images(self):
        input_dir = filedialog.askdirectory(title="Select Directory with Test Images")
        if not input_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Testing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Testing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                path_ = os.path.join(input_dir, filename)
                try:
                    original, field, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(path_)
                    self.results_logger.log_result_paired(path_, psnr_val, ssim_val)
                except Exception as e:
                    logging.error(f"Error processing {path_}: {e}")
                    continue

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            self.results_logger.save_metrics()
            progress_window.destroy()
            messagebox.showinfo("Success", "Batch testing complete!")

        except Exception as e:
            logging.error(f"Error in batch testing: {e}")
            messagebox.showerror("Error", f"Batch testing failed: {str(e)}")

    def view_test_results(self):
        summary = self.results_logger.get_summary()
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Test Results Summary")

        ttk.Label(summary_window, text=f"Average PSNR: {summary['psnr_avg']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"PSNR Std Dev: {summary['psnr_std']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"Average SSIM: {summary['ssim_avg']:.4f}").pack(pady=5)
        ttk.Label(summary_window, text=f"SSIM Std Dev: {summary['ssim_std']:.4f}").pack(pady=5)

        results_frame = ttk.Frame(summary_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        results_listbox = tk.Listbox(results_frame, yscrollcommand=scrollbar.set, width=100)
        for metric in self.results_logger.metrics:
            if metric['type'] == 'Paired':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | {os.path.basename(metric['image'])} | "
                    f"PSNR: {metric['psnr']:.2f} dB | SSIM: {metric['ssim']:.4f}"
                )
            elif metric['type'] == 'Standalone':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | Standalone Decode | {os.path.basename(metric['image'])} | Decoded Image"
                )
        results_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=results_listbox.yview)

    def batch_process(self):
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in input directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Batch Processing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Processing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = image.astype(float) / 255.0

                # For demonstration: compute band powers (assuming uniform band powers)
                band_powers = {
                    'theta': 1.0,
                    'alpha': 0.5,
                    'beta': 1.5,
                    'gamma': 2.0
                }
                # Generate and save field
                field_path = self.field_processor.save_field_from_original(image_path, band_powers)

                # Load the saved field
                if os.path.basename(field_path).startswith('field_'):
                    field_filename = os.path.basename(field_path)
                else:
                    field_filename = f"field_{os.path.basename(field_path)}"
                field_path = os.path.join(self.field_dir, field_filename)
                field = cv2.imread(field_path, cv2.IMREAD_GRAYSCALE)
                if field is None:
                    continue
                field = cv2.resize(field, (self.resolution, self.resolution))

                # Apply color map for visualization or other purposes
                colored = cv2.applyColorMap(field, cv2.COLORMAP_JET)

                # Save colored field to output directory
                output_path = os.path.join(output_dir, field_filename)
                cv2.imwrite(output_path, colored)

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", "Batch processing complete!")

        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")

    def start_recording(self):
        if self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Already recording!")
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not output_path:
            return
        try:
            self.video_recorder.start_recording(output_path)
            messagebox.showinfo("Recording", f"Recording started: {output_path}")
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        if not self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Not currently recording!")
            return
        try:
            self.video_recorder.stop_recording()
            messagebox.showinfo("Recording", "Recording stopped and saved.")
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                messagebox.showinfo("Success", "Model loaded successfully!")
                logging.info(f"Model loaded from {filepath}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def save_model(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                torch.save({'model_state_dict': self.model.state_dict()}, filepath)
                messagebox.showinfo("Success", "Model saved successfully!")
                logging.info(f"Model saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def train_model(self):
        """Train the U-Net model on original vs field images."""
        original_dir = filedialog.askdirectory(title="Select Original Images Directory")
        if not original_dir:
            return
        field_dir = filedialog.askdirectory(title="Select Field Images Directory")
        if not field_dir:
            return

        try:
            original_files = [
                f for f in os.listdir(original_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            # Assuming field images are named as 'field_<original_filename>'
            field_files = [f"field_{f}" for f in original_files]

            valid_pairs = []
            for orig, field in zip(original_files, field_files):
                if os.path.exists(os.path.join(field_dir, field)):
                    valid_pairs.append((field, orig))  # Order: field first, then original

            if not valid_pairs:
                messagebox.showerror("Error", "No matching image pairs found")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training Progress")
            progress_window.geometry("400x200")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Preparing training...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            num_epochs = 1200
            batch_size = 4
            learning_rate = 0.001

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            split_idx = int(0.8 * len(valid_pairs))
            train_pairs = valid_pairs[:split_idx]
            val_pairs = valid_pairs[split_idx:]

            train_dataset = ImagePairDataset(original_dir, field_dir, train_pairs, transform)
            val_dataset = ImagePairDataset(original_dir, field_dir, val_pairs, transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')

            scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

            for epoch in range(num_epochs):
                train_loss = 0.0
                self.model.train()

                for batch_idx, (field_imgs, original_imgs) in enumerate(train_loader):
                    field_imgs = field_imgs.to(self.device, non_blocking=True)
                    original_imgs = original_imgs.to(self.device, non_blocking=True)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        outputs = self.model(field_imgs)
                        loss = criterion(outputs, original_imgs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                    total_batches = len(train_loader)
                    batch_progress = 100 * (batch_idx + 1) / total_batches
                    progress_var.set((epoch + batch_idx / total_batches) / num_epochs * 100)
                    status_var.set(f"Epoch {epoch+1}/{num_epochs} - Training...")
                    progress_window.update()

                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for field_imgs, original_imgs in val_loader:
                        field_imgs = field_imgs.to(self.device, non_blocking=True)
                        original_imgs = original_imgs.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                            outputs = self.model(field_imgs)
                            loss = criterion(outputs, original_imgs)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss
                    }, 'best_model.pth')
                    logging.info(f"Saved best model with Val Loss: {val_loss:.4f}")

                status_var.set(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", f"Training complete!\nBest validation loss: {best_val_loss:.4f}")

        except Exception as e:
            logging.error(f"Error in training: {e}")
            messagebox.showerror("Error", f"Error in training: {str(e)}")

    # -----------------------------------------------------------------------------
    # Main Function
    # -----------------------------------------------------------------------------
def main():
    # Configure logging
    logging.basicConfig(
        filename='eegnoids_app.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    root = tk.Tk()
    app = EEGNoidsApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()
