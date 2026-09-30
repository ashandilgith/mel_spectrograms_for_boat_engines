# mel_spectrograms_for_boat_engines# Piranaware: Multimodal Marine Engine Diagnostic AI

Piranaware is a machine learning pipeline and diagnostic triage application designed to detect, classify, and troubleshoot mechanical marine engine failures using acoustic telemetry and LLM workflows.

For marine technicians and boat owners, the system categorizes faults (e.g., lean fuel bog, belt squeal, dropped cylinder) from uploaded audio recordings. An integrated LLM then provides a step-by-step repair protocol based on the ML acoustic inference and user-uploaded images.

For engineering evaluation, this repository demonstrates a multi-system AI architecture encompassing Digital Signal Processing (DSP), procedural data generation, Convolutional Neural Networks (CNNs), and stateful multimodal LLM integration.

---

##  System Architecture

The repository is decoupled into four isolated systems to separate data generation, processing, and inference.

* **System 1: Core ML Classifier & LLM Assistant**
* Converts raw `.wav` data into standardized Mel-spectrogram arrays ($128 \times 44 \times 1$).
* Trains and serves a 2D CNN for multi-class image classification.
* Hosts the Streamlit web application integrating the CNN inference timeline with an OpenAI GPT-4o multimodal diagnostic agent.


* **System 2: Automated Audio Extractor**
* A utility pipeline utilizing `yt-dlp` and `ffmpeg` to extract and trim diagnostic audio from public videos without re-encoding.


* **System 3: Edge Data Ingestion**
* Structured directories for manual smartphone recordings or piezo-electric contact microphone telemetry, categorized by fault class.


* **System 4: Procedural DSP Synthesizer**
* An acoustic modeling engine built with `NumPy` and `SciPy`. It generates synthetic training data (simulating combustion pulses, transient misfire shockwaves, and harmonic friction) to train the CNN without requiring large-scale real-world datasets.



##  Tech Stack

* **Deep Learning & DSP:** TensorFlow / Keras, Librosa, SciPy, NumPy
* **Agentic LLM Integration:** OpenAI API (GPT-4o Vision & Chat)
* **Application & UI:** Streamlit
* **Infrastructure:** GitHub Codespaces (Devcontainers), Docker, Google Cloud Run

---

##  Machine Learning Pipeline

### 1. Procedural Data Synthesis

System 4 utilizes procedural generation to address data scarcity. It builds mathematical acoustic models of 8 engine states, applying lowpass filters (simulating engine cowlings) and Gaussian noise (simulating ambient water/wind interference) to create varied training data.

### 2. Feature Engineering

The pipeline processes variable-length audio by slicing it into 1-second chunks. Each chunk is processed by `librosa` into a 128-band Mel-spectrogram, converted to the decibel scale, and Min-Max normalized. This standardizes the input for the neural network to output a temporal timeline of faults.

### 3. The CNN Classifier

The model is a 2D Convolutional Neural Network utilizing:

* Sequential `Conv2D` layers (32, 64, 128 filters) with ReLU activation.
* `BatchNormalization` to standardize inputs across variable acoustic samples.
* `GlobalAveragePooling2D` to reduce spatial dimensions and mitigate overfitting on specific temporal locations of sound anomalies.
* A `Softmax` dense output layer mapping to the 8 diagnostic classes.

---

## 🗂 Diagnostic Taxonomy (Classes)

The model classifies the following physical failure modes based on spectral signatures:

1. `normal`: Baseline harmonic operation.
2. `belt_pulley_squeal`: Seized accessory pulleys or slipping belts (high-frequency continuous bands).
3. `lean_bog_starvation`: Clogged carburetor jets or fuel line restrictions (intake rush followed by harmonic frequency collapse).
4. `four_stroking_rich`: Stuck carburetor floats causing flooded cylinders (halved fundamental frequency with sub-400Hz energy).
5. `misfire_backfire`: Out-of-sync timing or crossed plug wires (full-spectrum vertical transient spikes).
6. `cylinder_drop`: Dead spark plugs or zero compression (asymmetric, missing harmonic tracks).
7. `rpm_ceiling_bog`: Fuel pressure drops at high engine speeds (harmonic frequency climb flattens prematurely).
8. `start_and_die`: Priming pressure loss (brief burst followed by total spectral collapse).

---

##  Installation & Execution

This repository includes a `.devcontainer` configuration for execution in GitHub Codespaces, which pre-installs the necessary C-libraries (`libsndfile1`, `ffmpeg`).

**1. Generate the Procedural Training Data**

```bash
python system_4_synthesizer/run_synthesis.py --mode synthesize --count 25

```

**2. Convert Audio to Mel-Spectrograms**

```bash
python system_1_classifier/audio_to_mel.py

```

**3. Train the CNN Classifier**

```bash
python system_1_classifier/train_classifier.py

```

**4. Launch the Multimodal Triage Hub**
Ensure the `.env` file contains the `OPENAI_API_KEY`.

```bash
streamlit run app.py

```