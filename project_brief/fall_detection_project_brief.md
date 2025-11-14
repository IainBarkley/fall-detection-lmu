# Fall Detection Research Project Brief

## Project Overview
**Goal**: Implement and compare a traditional CNN-based fall detection algorithm with a spiking neural network (SNN) version to evaluate potential advantages in latency, power consumption, accuracy, and memory usage for pre-impact fall detection systems.

## Research Methodology

### Part 1: Algorithm Implementation and Validation
1. **CNN Implementation**: Replicate the fall detection algorithm from Jain & Semwal (2022) paper
   - Implement the CNN-LSTM ensemble model ("FallNet")
   - Achieve reported performance metrics (99.24% sensitivity, 98.79% F1-score)
   - Validate on SisFall and KFall datasets with dataset fusion approach

2. **SNN Implementation**: Convert the validated CNN model to spiking neural network
   - Use Legendre polynomials for temporal encoding
   - Match or exceed CNN performance metrics
   - Implement rate coding and temporal dynamics

3. **Performance Comparison**: Evaluate both systems on:
   - **Accuracy**: Sensitivity, specificity, F1-score
   - **Latency**: Inference time and reaction time (target: <0.5s lead time)
   - **Power**: Energy consumption per inference
   - **Memory**: Model size and runtime memory usage

### Part 2: Embedded System Deployment
If Part 1 is successful, deploy both models on embedded hardware to measure:
- Real-world inference time
- Power draw during operation
- Memory footprint
- Practical deployment feasibility

## Technical Details from Source Papers

### Primary Algorithm (Jain & Semwal 2022)
- **Architecture**: CNN-LSTM ensemble (FallNet)
  - 1D CNN with 128 filters (3x filter size), MaxPooling
  - LSTM with 256 units
  - Dense layers: 128→64→32→8 (softmax)
  - Input: (200, 6) - 1 second of tri-axial acc + gyro data
  - 8-class classifier: Walking, Jogging, Stairs, Stumble, Fall Recovery, Fall Initiation, Impact, Aftermath

- **Novel Feature Extraction**:
  - Automatic temporal segmentation using statistical measures
  - Window size: 0.25s (50 samples at 200Hz)
  - Transitional window (Tw) concept for early detection
  - Segmentation based on standard deviation peaks in Y-axis acceleration

- **Key Performance Metrics**:
  - Sensitivity: 99.24% (Fall Initiation detection)
  - F1-score: 98.79%
  - Overall accuracy: 97.52%
  - Lead time: 0.5s (reaction time)

### Dataset Information
**Source Datasets:**
- **SisFall**: Comprehensive dataset (38 subjects, 19 ADLs + 15 falls, 200Hz sampling) - used for scale and diversity
- **KFall**: Temporal labeled dataset (32 Korean males, 21 ADLs + 15 falls, 100Hz) - provides phase labeling

**Your Current Dataset:**
- **Structure**: SA06-SA38 merged processed data (32 subjects)
- **Format**: 126,436 samples × 14 columns per subject file
- **Columns**: TimeStamp, FrameCounter, AccX/Y/Z, GyrX/Y/Z, EulerX/Y/Z, Fall/No Fall, Fall Type
- **Sampling Rate**: 100Hz (0.01s intervals)
- **Labels**: Binary (Fall/No Fall) + Multi-class (Fall Type)
- **Data Fusion**: Both datasets combined following Jain & Semwal methodology

### SNN Conversion Strategy
- **Framework**: Nengo (selected)
- **Temporal Processing**: Legendre Memory Units (LMUs) for temporal feature encoding
- **Architecture**: Convert CNN-LSTM to spiking equivalent using LMU layers
- **Training**: Direct SNN training with rate coding

## Research Questions
1. Can SNN achieve comparable accuracy to traditional CNN for fall detection?
2. What are the energy efficiency gains of SNN vs CNN on embedded hardware?
3. How does inference latency compare between architectures?
4. What is the memory footprint difference?
5. Which approach is more suitable for real-time wearable applications?

## Technical Challenges
- **Sampling Rate Alignment**: Your data is 100Hz but Jain & Semwal used 200Hz inputs - need to address this difference
- **Label Mapping**: Verify your Fall Type labels map to their 8-class taxonomy (Walking, Jogging, Stairs, Stumble, Fall Recovery, Fall Initiation, Impact, Aftermath)
- **Dataset Preprocessing**: Ensure z-score standardization and data fusion matches their methodology exactly
- **LMU Architecture Design**: Converting CNN-LSTM temporal dynamics to LMU-based spiking equivalents
- **Hardware Deployment**: Implementing both architectures on embedded systems with accurate power/performance measurement
- **Performance Validation**: Achieving 99.24% sensitivity target with SNN while maintaining sub-0.5s reaction time

## Expected Outcomes
- Validated CNN implementation matching paper results
- Novel SNN architecture for fall detection
- Comprehensive performance comparison
- Guidelines for choosing between CNN/SNN for wearable fall detection
- Potential publication on neuromorphic computing for healthcare applications

## Timeline Considerations
- **Phase 1** (CNN replication): 2-3 months
- **Phase 2** (SNN development): 3-4 months  
- **Phase 3** (Embedded evaluation): 2-3 months
- **Total estimated duration**: 7-10 months

## Hardware Requirements
- **Development**: High-performance workstation with GPU
- **Embedded Testing**: Raspberry Pi, Jetson Nano, or neuromorphic chips (Loihi, SpiNNaker)
- **Sensors**: IMU modules for validation testing
- **Power Measurement**: Equipment for accurate power consumption analysis