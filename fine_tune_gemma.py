import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import os
import psutil


# Function to measure memory
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    print(f"[{step}] RAM Usage: {process.memory_info().rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"[{step}] VRAM Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma with Zenith optimization"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b",
        help="Model name (e.g., google/gemma-2b)",
    )
    parser.add_argument(
        "--use_zenith", action="store_true", help="Enable Zenith optimization"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    print(f"Starting Fine-Tuning Task with Model: {args.model_name}")
    print(f"Zenith Optimization: {'ENABLED' if args.use_zenith else 'DISABLED'}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Fix for Gemma/Llama

    # Load Model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load in fp16 or bf16 if available
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            # token=os.getenv("HF_TOKEN") # Optional: explicit token
        )
    except OSError as e:
        if "gated" in str(e) or "401" in str(e):
            print(f"\nERROR: Access to model '{args.model_name}' is restricted/gated.")
            print(
                "Please ensure you are logged in using `huggingface-cli login` or set HF_TOKEN environment variable."
            )
            print("Running `huggingface-cli login` now might fix this.")
            return
        raise e

    # Apply LoRA to make it trainable on consumer GPUs (optional but recommended for 2B+)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ZENITH INTEGRATION
    if args.use_zenith:
        print("\n--- INITIATING ZENITH OPTIMIZATION ---")
        try:
            import zenith

            print(f"Zenith version: {zenith.__version__}")

            # Hypothetical Zenith API Usage:
            # 1. Compile the model for training
            # Note: torch.compile/zenith usually optimizes the forward pass.

            print("Applying Zenith backend to model...")
            # We compile the underlying model, not the PeftModel wrapper directly if possible,
            # or compile the forward pass.
            # Using torch.compile with zenith backend
            model.model = torch.compile(model.model, backend="zenith")

            print("✓ Zenith Compilation Enabled")
        except ImportError:
            print("CRITICAL: 'zenith' library not found! Optimization skipped.")
        except Exception as e:
            print(f"Error applying Zenith: {e}")

    # Dataset (Dummy dataset for speed test or small real one)
    print("Loading dataset...")
    # Use a small subset of alpaca
    dataset = load_dataset(
        "tatsu-lab/alpaca", split="train[:100]"
    )  # Small subset for testing

    def format_prompt(sample):
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        max_steps=50,  # Limit steps for testing speed
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=format_prompt,
        max_seq_length=512,
    )

    print("\nStarting Training Loop...")
    print_memory_usage("Pre-Train")

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print_memory_usage("Post-Train")
    print(f"\nTraining Complete!")
    print(f"Total Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
