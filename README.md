# Protein Mutation Prediction Pipeline

A comprehensive, automated, and generalizable pipeline for predicting amino acid mutations in proteins using recurrent neural networks. Originally designed for SARS-CoV-2 Spike protein, the pipeline now supports any protein with minimal configuration changes. This research focuses on understanding temporal mutation patterns through advanced machine learning techniques.

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools

# 2. Run the complete pipeline
python main.py --config configs/sars_cov_2_default.yaml full-pipeline

# 3. Or run individual steps
python main.py --config configs/sars_cov_2_default.yaml prepare    # Data preparation
python main.py --config configs/sars_cov_2_default.yaml cluster    # Sequence clustering  
python main.py --config configs/sars_cov_2_default.yaml dataset    # Dataset creation
python main.py --config configs/sars_cov_2_default.yaml train      # Model training
```

### Docker Deployment

```bash
# Build and run complete pipeline
bash docker/main.sh --full

# Run specific pipeline steps
bash docker/main.sh --prepare
bash docker/main.sh --train

# Build images only
bash docker/main.sh --build-only
```

## 📁 Project Structure

```
bachelor_thesis/
├── configs/                    # Configuration files
│   └── sars_cov_2_default.yaml # Default pipeline configuration
├── data/                       # Data directory
│   ├── input/                  # Raw FASTA files and ProtVec embeddings
│   └── processed/              # Processed datasets and intermediate files
├── docker/                     # Docker deployment files
│   ├── base.Dockerfile         # Base image with dependencies
│   ├── Dockerfile              # Application image
│   └── main.sh                 # Docker deployment script
├── models/                     # Trained model outputs
├── results/                    # Training results and visualizations
├── scripts/                    # Source code
│   ├── config.py               # Configuration loader
│   ├── utils.py                # Shared utilities
│   ├── pipeline/               # Pipeline modules
│   │   ├── prepare_data.py     # Data preprocessing
│   │   ├── create_clusters.py  # K-means clustering
│   │   ├── link_clusters.py    # Temporal cluster linking
│   │   ├── create_dataset.py   # Dataset creation
│   │   └── train_model.py      # Training orchestrator
│   └── training/               # Training components
│       ├── architectures.py    # Neural network models
│       ├── trainer.py          # Training loop
│       ├── evaluator.py        # Model evaluation
│       ├── visualizer.py       # Result visualization
│       └── dataset_processor.py # Data loading and splitting
├── main.py                     # Pipeline orchestrator
└── README.md                   # This file
```

## 🔧 Configuration

The pipeline is fully configurable through YAML files and supports both **automated** and **manual** modes for maximum flexibility. The default configuration (`configs/sars_cov_2_default.yaml`) includes:

- **Data paths**: Input files, intermediate processing directories, output locations
- **Automated features**: Auto-detection of sequence lengths, automatic optimal cluster determination
- **Pipeline parameters**: Clustering settings, epitope definitions, window sizes
- **Model configuration**: Architecture selection, hyperparameters, training settings
- **Evaluation settings**: Metrics, visualization options, model saving strategies

### 🤖 Automation Features

- **Automatic Cluster Optimization**: Uses Silhouette Score to find optimal number of clusters
- **Auto-Detection of Sequence Lengths**: No need to manually specify expected sequence lengths
- **Generalized Protein Support**: Easy adaptation to proteins other than SARS-CoV-2

### Key Configuration Sections

```yaml
# Automated clustering
cluster:
  auto_k_selection:
    enabled: true    # Enable automatic cluster optimization
    min_k: 2         # Minimum number of clusters to test
    max_k: 15        # Maximum number of clusters to test
  
# Automated sequence processing
prepare:
  fasta_prefix: "batch_data"  # Configurable filename prefix
  clean_sequences:
    expected_len: null        # Auto-detect from data (set to number for manual mode)
    error_margin: 13          # Length variation tolerance

# Model selection and training
train:
  model_type: 'AttentionRNN'  # 'RNN', 'AttentionRNN', 'DualAttentionRNN', or 'custom'
  hyperparameters:
    AttentionRNN:
      hidden_size: 256
      dropout: 0.2
      learning_rate: 0.0005
      batch_size: 256
      epochs: 200
```

## 🧬 Pipeline Overview

### 1. Data Preparation (`--prepare`)
- Processes raw FASTA files containing SARS-CoV-2 spike protein sequences
- Filters sequences by length and quality criteria
- Organizes sequences by time periods (monthly/quarterly)
- Generates cleaned CSV files for downstream processing

### 2. Sequence Clustering (`--cluster`)
- Transforms amino acid sequences to ProtVec embeddings (100-dimensional vectors)
- Applies K-means clustering within each time period
- Uses predetermined cluster numbers optimized for temporal consistency
- Supports multiprocessing for efficient computation

### 3. Temporal Cluster Linking (`--link`)
- Links clusters across consecutive time periods
- Uses Euclidean distance between cluster centroids
- Maintains temporal coherence of sequence evolution
- Creates linked cluster trajectories for dataset construction

### 4. Dataset Creation (`--dataset`)
- Extracts epitope regions from linked cluster sequences
- Creates sliding window datasets (default: 10 time periods)
- Implements similarity thresholding to filter inconsistent sequences
- Balances mutation/non-mutation samples for training

### 5. Model Training (`--train`)
- **Flexible Architecture Support**: Choose from RNN, AttentionRNN, DualAttentionRNN, or custom models
- **Comprehensive Metrics**: Accuracy, precision, recall, F-score, Matthews Correlation Coefficient (MCC)
- **Smart Model Saving**: Save best models or models meeting performance thresholds with unique names
- **Baseline Comparison**: Optional logistic regression baseline for performance comparison
- **Rich Visualization**: Training curves, attention weights, ROC curves, confusion matrices

### 6. Model Evaluation (`--test`)
- Evaluates trained models on held-out test sets
- Generates comprehensive performance reports
- Creates publication-ready visualizations
- Compares against baseline methods

## 🤖 Model Architectures

### Available Models

1. **RNN**: Basic recurrent neural network for temporal sequence modeling
2. **AttentionRNN**: RNN with attention mechanism for focusing on relevant time periods
3. **DualAttentionRNN**: Advanced model with dual attention (temporal + feature-level)
4. **Custom Models**: Support for user-defined architectures

### Model Features

- **Temporal Attention**: Focus on relevant time periods for prediction
- **Feature Attention**: Highlight important amino acid positions
- **Dropout Regularization**: Prevent overfitting in complex models
- **Configurable Architecture**: Adjust hidden sizes, layers, and hyperparameters

## 📊 Key Features

### Advanced Evaluation Metrics
- **Matthews Correlation Coefficient (MCC)**: Primary metric for imbalanced datasets
- **ROC-AUC Analysis**: Comprehensive receiver operating characteristic curves
- **Precision/Recall/F-Score**: Standard classification metrics
- **Confusion Matrices**: Detailed prediction analysis

### Intelligent Model Management
- **Unique Model Naming**: Automatic timestamping and performance-based naming
- **Flexible Saving Strategies**: Save best models, threshold-based, or both
- **Model Metadata**: Complete hyperparameter and training configuration storage
- **Performance Tracking**: Detailed training history and metrics logging

### Production-Ready Features
- **Docker Deployment**: Containerized pipeline for reproducible experiments
- **Configuration Management**: YAML-based configuration with validation
- **Comprehensive Logging**: Detailed execution logs for debugging and monitoring
- **Error Handling**: Robust error management and recovery

## 🧪 Research Background

### Biological Context
- **SARS-CoV-2 Spike Protein**: Critical protein for viral entry into host cells
- **Epitope Regions**: Specific amino acid sequences recognized by immune system
- **Temporal Evolution**: Understanding how mutations occur over time
- **Prediction Applications**: Early warning systems for variant emergence

### Technical Approach
- **ProtVec Embeddings**: 100-dimensional vector representations of amino acid triplets
- **Temporal Clustering**: K-means clustering preserving time-series structure
- **RNN Architecture**: Specialized for sequence-to-sequence prediction
- **Attention Mechanisms**: Focus on relevant temporal and spatial features

## 📈 Usage Examples

### Custom Configuration
```bash
# Create custom config file
cp configs/sars_cov_2_default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your parameters

# Run with custom config
python main.py --config configs/my_experiment.yaml --train
```

### Model Selection
```yaml
# In your config file
train:
  model_type: 'DualAttentionRNN'
  hyperparameters:
    DualAttentionRNN:
      hidden_size: 512
      dropout: 0.3
      learning_rate: 0.001
      epochs: 300
```

### Docker Deployment with Custom Config
```bash
# Use custom configuration
bash docker/main.sh --config configs/my_experiment.yaml --train
```

## 🔬 Development

### Code Formatting
```bash
black .  # Format Python code
```

### Testing
```bash
# Test individual pipeline steps
python -m scripts.pipeline.prepare_data
python -m scripts.pipeline.create_clusters
```

### Adding Custom Models
1. Define your model in `scripts/training/architectures.py`
2. Update the `create_model()` function to include your architecture
3. Add hyperparameters to your configuration file
4. Set `model_type: 'YourModelName'` in config

## 📄 Dependencies

### Core Requirements
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities  
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization and plotting
- **PyYAML**: Configuration file parsing
- **numpy**: Numerical computing

### Development Tools
- **black**: Code formatting
- **pytest**: Testing framework (optional)

## 🚨 Important Notes

### Data Requirements
- Input FASTA files should be placed in `data/input/`
- ProtVec embeddings file: `protVec_100d_3grams.csv` required (must use **tab-separated** format)
- Sufficient disk space for intermediate processing files

### Performance Considerations
- K-means clustering can be memory-intensive for large datasets
- GPU acceleration recommended for training (PyTorch CUDA support)
- Multiprocessing enabled for cluster creation (adjust based on available cores)

### Configuration Best Practices
- Always validate configuration files before running
- Use appropriate cluster numbers for your dataset size
- Monitor training metrics and adjust hyperparameters accordingly
- Set realistic MCC thresholds for model saving

## 📚 Research Applications

This pipeline supports various research applications:

- **Variant Prediction**: Early detection of emerging SARS-CoV-2 variants
- **Epitope Evolution**: Understanding immune escape mechanisms
- **Temporal Analysis**: Tracking mutation patterns over time
- **Comparative Studies**: Analyzing different viral proteins or organisms

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Update configuration schema for new parameters
4. Test changes with sample datasets
5. Update documentation as needed

## 📞 Support

For issues, questions, or contributions:
- Check the configuration file format and required data files
- Review logs for detailed error information
- Ensure all dependencies are properly installed
- Verify Docker setup for containerized deployment

---

**Note**: This pipeline is designed for research purposes in computational biology and bioinformatics. Results should be validated with domain expertise and additional experimental validation.