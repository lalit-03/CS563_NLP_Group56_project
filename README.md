# Empathetic Chatbot with Conversation Summarization

This repository contains code and resources for an empathetic chatbot system that maintains context in conversations through advanced summarization capabilities. 

The project was developed by Group 56:
- Devendra Pratap Singh(2101AI13)
- Lalit Chandra Routhu(2101AI17)
- Tejas Budhwal(2101AI42)

## Project Overview

Our empathetic chatbot system uses a sophisticated approach to understand and respond to users with emotional intelligence. By leveraging conversation summarization, the chatbot maintains context throughout interactions, enabling more personalized and relevant responses even in extended conversations.

The system consists of three main components:

- Dataset generation using GPT-4o
- Fine-tuning Gemma3 for conversation summarization
- Deploying a chatbot that uses the summarization model to maintain context

## Repository Contents

- `dataset_generation_gpt4o.ipynb`: Generates a dataset of conversation summaries using GPT-4o and the EmpatheticDialogue dataset[1]
- `train_gemma3_summarization.ipynb`: Fine-tunes a Gemma3-1B model on conversation summarization tasks[3]
- `run_chatbot.ipynb`: Implements and runs the empathetic chatbot with the trained summarization model[2]
- `all_summaries.json`: Dataset containing conversations and their summaries for training[4]

## Technical Implementation

### Dataset Generation

The dataset generation process extracts conversations from the EmpatheticDialogue dataset and generates summaries using OpenAI's GPT-4o model. The script:

- Loads the EmpatheticDialogue dataset
- Extracts and formats conversations
- Generates concise summaries that capture emotional context
- Saves the data in a structured JSON format[1]

### Model Training

We fine-tune Google's Gemma3-1B-Instruct model for the summarization task using:

- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 8-bit quantization for memory efficiency
- 3 epochs of training on 800 conversation examples
- Evaluation on a held-out set of 200 examples[3]

### Chatbot Implementation

The chatbot operates using a two-stage pipeline:

1. **Summarization**: The fine-tuned Gemma3 model generates a concise summary of the conversation history
2. **Response Generation**: A separate model generates empathetic responses based on the summary and the latest user message[2]

## Setup and Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PEFT, TRL, Accelerate libraries
- OpenAI API key (for dataset generation)
- GPU access (recommended for training and inference)

### Dataset Generation

```python
# Set your OpenAI API key
API_KEY = "your-openai-api-key"

# Configure parameters
MAX_EXAMPLES = 1000  # Number of examples to summarize
BATCH_SIZE = 20      # Process in batches
MAX_WORKERS = 5      # Parallel workers

# Run the dataset generation notebook
# This will create all_summaries.csv and all_summaries.json
```

### Training the Summarization Model

```python
# Set up the model and training configuration
model_id = "google/gemma-3-1b-it"
output_dir = "./gemma3-1b-summarization-finetuned"
max_seq_length = 1024

# Run the training notebook
# This will fine-tune the model and save the adapter weights
```

### Running the Chatbot

```python
# Configure the models
SUMMARIZER_ADAPTER_ID = "your-adapter-path-or-huggingface-id"
BASE_MODEL_ID = "google/gemma-3-1b-it"

# Run the chatbot
# Interact through the provided interface
```

## Example Conversation

The chatbot can maintain context across multiple turns of conversation, providing empathetic responses that acknowledge user emotions:

User: "I just moved to a new city for work and I'm feeling really lonely."

Bot: _[Generates summary internally]_

Bot: "Moving to a new city can be really challenging, especially when you don't know anyone yet. It's completely normal to feel lonely in this situation. Have you had a chance to explore your new neighborhood or join any local groups that match your interests?"

## Future Work

- Enhance summarization capabilities for longer conversations
- Implement better emotion recognition
- Add multilingual support
- Optimize for mobile deployment

## Contributors

Group 56 Team Members:

- Lalit Chandra Rothu (2101AI17)
- Devendra Pratap Singh (2101AI13)
- Tejas Budhwal (2101AI42)
