# Import necessary libraries (Install if needed)
import os

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names # Custom package, make sure the .py file is in the working directory

from train_args_jihyun import ScriptArguments # Custom package, make sure the .py file is in the working directory

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

# Parse script arguments using Hugging Face's argument parser
parser = HfArgumentParser(ScriptArguments) # ScriptArguments is a Custom package available in train_args_jihyun.py
# Convert parsed arguments into data classes
args = parser.parse_args_into_dataclasses()[0]

def training_function(args): # Define the training function
    # Log in to Hugging Face Hub using the provided token
    login(token=args.hf_token)
     # Set the seed for reproducibility
    set_seed(args.seed)
    # Store the data path from arguments
    data_path=args.data_path
    # Load dataset
    dataset = load_dataset(data_path)
    # Configure BitsAndBytes for efficient model quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # Enable loading model in 4-bit
        bnb_4bit_use_double_quant=True, # Use double quantization for 4-bit
        bnb_4bit_quant_type="nf4", # Set quantization type to nf4
        bnb_4bit_compute_dtype=torch.bfloat16, # Set computation data type to bfloat16
    )

    # Load the model with QLoRA configuration for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False, # Disable caching
        device_map="auto", # Automatically distribute model across available devices
        quantization_config=bnb_config, # Apply the quantization configuration
        trust_remote_code=True # Trust code from Hugging Face Hub
    )

    # Load tokenizer for the model and configure padding
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token=tokenizer.eos_token # Use end-of-sequence token for padding
    tokenizer.padding_side='right' # Pad on the right side
    # Prepare model for training with k-bit precision
    model=prepare_model_for_kbit_training(model)
    # Find all linear layers in the model for LoRA adaptation
    modules=find_all_linear_names(model)
    # Configure LoRA parameters
    config = LoraConfig(
        r=64, # LoRA attention dimension
        lora_alpha=16, # Alpha parameter for LoRA scaling
        lora_dropout=0.1, # Dropout probability for LoRA layers
        bias='none', # No bias in LoRA layers
        task_type='CAUSAL_LM', # Specify task type for LoRA adaptation
        target_modules=modules # Target modules for adaptation
    )

    model=get_peft_model(model, config) # Apply LoRA configuration to the model
    output_dir = args.output_dir
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir, # Directory to save output
        per_device_train_batch_size=args.per_device_train_batch_size, # Batch size per device
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Steps for gradient accumulation
        optim=args.optim, # Optimizer choice
        save_steps=args.save_steps, # Steps interval to save model checkpoint
        logging_steps=args.logging_steps, # Steps interval for logging
        learning_rate=args.learning_rate, # Learning rate for optimizer
        bf16=False, # Disable bf16 precision
        max_grad_norm=args.max_grad_norm, # Maximum gradient norm for clipping
        num_train_epochs=args.num_train_epochs, # Total number of training epochs
        warmup_ratio=args.warmup_ratio, # Ratio for learning rate warmup
        group_by_length=True, # Group sequences by length
        lr_scheduler_type=args.lr_scheduler_type, # Type of learning rate scheduler
        tf32=False, # Disable tf32 precision
        report_to="none", # Disable reporting to Hugging Face Hub
        push_to_hub=False, # Disable automatic push to Hugging Face Hub
        max_steps = args.max_steps # Maximum number of training steps
    )
    
    # Initialize the trainer with LoRA model and training configurations
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'].select(range(2000)), # Subsets the first 2000 observations of the training dataset
        dataset_text_field=args.text_field, # Text field in dataset
        max_seq_length=2048, # Maximum sequence length
        tokenizer=tokenizer, # Tokenizer for processing text
        args=training_arguments # Training arguments
    )

    # Convert layer normalization layers to float32 precision for stability
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    # Start the training
    trainer.train()

    print('LoRA training complete')
    lora_dir = args.lora_dir

    # Push trained LoRA adapters to Hugging Face Hub
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    
# Execute the training function if this script is run as the main program
if __name__=='__main__':
    training_function(args)
