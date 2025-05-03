# GPT Model Implementation and Training

This project demonstrates the implementation, training, and usage of a Generative Pre-trained Transformer (GPT) model using PyTorch. It includes steps for building the model architecture, preparing data, training from scratch, evaluating performance, generating text, and loading pre-trained weights from OpenAI's GPT-2.

## Key Features & Components

* **GPT Model Architecture (`scaffolding.py`, `.ipynb`):**
    * Implementation of a decoder-only transformer (`GPTModel`).
    * Core components: `MultiHeadAttention` (with causal masking), `FeedForward` networks, `LayerNorm`, `GELU` activation.
    * Token (`nn.Embedding`) and Positional Embeddings (`nn.Embedding`).
* **Data Handling (`scaffolding.py`, `.ipynb`):**
    * Uses `tiktoken` for GPT-2 compatible tokenization.
    * Custom `GPTDatasetV1` and `create_dataloader_v1` for efficient batching and preparation of input/target sequences from raw text.
    * Splits data into training and validation sets.
* **Training & Evaluation (`.ipynb`):**
    * Defines functions for calculating batch loss (`calc_loss_batch`) and loader loss (`calc_loss_loader`) using Cross-Entropy.
    * Implements a training loop (`train_model_simple`) with:
        * `AdamW` optimizer.
        * Gradient calculation and weight updates.
        * Periodic evaluation on training and validation sets (`evaluate_model`).
    * Tracks and plots loss curves (`plot_losses`) against epochs and tokens seen using `matplotlib`.
    * Calculates Perplexity score based on validation loss.
* **Text Generation (`.ipynb`, `scaffolding.py`):**
    * Basic text generation using greedy decoding (`generate_text_simple`).
    * Advanced text generation (`generate`) incorporating **temperature scaling** and **top-k sampling** for controlled randomness.
* **Model Checkpointing (`.ipynb`):**
    * Saves trained model weights (`model_state_dict`) and optimizer state (`optimizer_state_dict`) using `torch.save`.
    * Loads saved checkpoints using `torch.load`.
* **Loading Pre-trained Weights (`.ipynb`):**
    * Includes functionality (`gpt_download`, `load_weights_into_gpt`) to download official OpenAI GPT-2 model weights (e.g., "124M").
    * Maps and assigns the downloaded weights to the corresponding layers in the custom `GPTModel` implementation.
    * Demonstrates text generation using the model loaded with pre-trained weights.

## Demonstrated Workflow in Notebook

1.  **Model Definition:** Define the GPT model architecture and configuration (`GPT_CONFIG_124M`).
2.  **Untrained Generation:** Show sample output from the randomly initialized model.
3.  **Data Loading:** Download sample text data (`the-verdict.txt`), tokenize, split, and create `DataLoader` instances.
4.  **Initial Loss:** Calculate the initial training and validation loss before training.
5.  **Training:** Train the model on the sample text for a defined number of epochs, evaluating periodically and generating samples.
6.  **Evaluation:** Plot the training and validation loss curves and calculate perplexity.
7.  **Trained Generation:** Generate text using the custom-trained model (both greedy and sampling methods).
8.  **Save/Load:** Save the trained model and optimizer state, then load them back into a new model instance.
9.  **Load OpenAI Weights:** Download GPT-2 ("124M") weights, instantiate a compatible model, and load the pre-trained weights.
10. **Pre-trained Generation:** Generate text using the model loaded with official OpenAI weights.
