from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:

    hf_token: str = field(metadata={"help": "A token information by Hugging Face, used for authentication to access Hugging Face's Model Hub and other services."})


    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "The identifier for the pre-trained model you intend to use from Hugging Face's Model Hub."}
    )

    seed: Optional[int] = field(
        default=4761, metadata = {'help':'A seed value for random number generators, ensuring reproducibility of your results'}
    )

    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": "The file path to your training dataset."}
    )

    output_dir: Optional[str] = field(
        default="output", metadata={"help": "The directory where the trained model, its configurations, and any additional output should be saved."}
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":"The number of training examples used in one forward/backward pass per training device. This affects memory consumption and training speed."}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":"The number of forward/backward passes to accumulate gradients before performing an optimization step."}
    )

    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":"The optimizer used for training. paged_adamw_32bit refers to a variation of the AdamW optimizer optimized for 32-bit."}
    )

    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":"How often to save the model checkpoint. When its value is set at 25, the model and optimizer state will be saved every 25 training steps."}
    )

    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":"The frequency at which training progress is logged."}
    )

    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":"The initial learning rate for the optimizer. It's a crucial hyperparameter affects how much the model weights are adjusted during training."}
    )

    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":"The maximum norm of the gradients for gradient clipping. Gradient clipping is used to prevent the exploding gradient problem by limiting the size of the gradients."}
    )

    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":"The total number of passes over the entire training dataset. More epochs can lead to better training at the risk of overfitting."}
    ) 

    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":"The proportion of training steps to perform linear learning rate warmup. During warmup, the learning rate gradually increases from 0 to the initially specified learning rate."}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":"The learning rate scheduler type, which adjusts the learning rate based on the number of epochs or steps. The cosine suggests a cosine decay learning rate schedule."}
    ) 

    lora_dir: Optional[str] = field(default = "./model/llm_hate_speech_lora", metadata = {"help":"Directory for saving any LoRA-specific model adjustments or configurations, if applicable."})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "The maximum number of training steps to execute. Training stops when this number is reached. If set to -1, training will continue until the number of epochs is reached."})

    text_field: Optional[str] = field(default='chat_sample', metadata={"help": "The name of the field in your dataset that contains the text to be processed. This is important for the model to know which part of your data is the input text."})


