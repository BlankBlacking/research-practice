import os
import json
import tempfile
import logging
import openai
from openai import OpenAI
from datetime import datetime

from src.common import load_from_jsonl
from src.models.common import num_tokens_gpt3
from src.models.openai_complete import get_cost_per_1k_tokens

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")


def convert_to_chat_format(input_path: str, output_path: str) -> None:
    """Convert old prompt/completion format to new messages format for gpt-3.5-turbo/gpt-4.
    
    Old format: {"prompt": "...", "completion": "..."}
    New format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    examples = load_from_jsonl(input_path)
    converted_examples = []
    
    for example in examples:
        if "messages" in example:
            # Already in new format
            converted_examples.append(example)
        else:
            # Convert from prompt/completion format
            prompt = example.get("prompt", "").strip()
            completion = example.get("completion", "").strip()
            
            converted = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            }
            converted_examples.append(converted)
    
    # Save converted data
    with open(output_path, "w") as f:
        for example in converted_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✅ Converted {len(converted_examples)} examples to messages format")
    print(f"   Saved to: {output_path}")


def submit_openai_finetune(
    model: str,
    training_path: str,
    validation_path: str,
    n_epochs: int,
    learning_rate_multiplier: float,
    batch_size: int,
    dataset_name: str,
):
    """Submit a finetune job to OpenAI using Python API."""
    import time as time_module
    
    # Convert backslashes to forward slashes for cross-platform compatibility
    training_path = training_path.replace("\\", "/")
    validation_path = validation_path.replace("\\", "/")
    
    print(f"\n{'='*60}")
    print(f"📋 Fine-tuning Configuration")
    print(f"{'='*60}")
    print(f"Training file: {training_path}")
    print(f"Validation file: {validation_path}")
    print(f"Model: {model}")
    
    # Map old model names - use gpt-3.5-turbo
    model_mapping = {
        "ada": "gpt-3.5-turbo",
        "babbage": "gpt-3.5-turbo",
        "curie": "gpt-3.5-turbo",
        "davinci": "gpt-3.5-turbo"
    }
    actual_model = model_mapping.get(model, model)
    
    # Always use messages format for gpt-3.5-turbo
    print(f"\n🔄 Using model: {actual_model}")
    print(f"   Format: messages/chat (required for gpt-3.5-turbo)")
    
    # Convert original files to messages format
    print(f"📝 Converting data to messages format...")
    
    data_dir = os.path.dirname(training_path)
    train_messages_path = os.path.join(data_dir, "train_messages.jsonl")
    val_messages_path = os.path.join(data_dir, "val_messages.jsonl")
    
    # Convert if not already done
    if not os.path.exists(train_messages_path):
        print(f"   Converting training data...")
        convert_to_chat_format(training_path, train_messages_path)
    else:
        print(f"   ✓ Using existing converted training file")
    
    if not os.path.exists(val_messages_path):
        print(f"   Converting validation data...")
        convert_to_chat_format(validation_path, val_messages_path)
    else:
        print(f"   ✓ Using existing converted validation file")
    
    train_file_to_upload = train_messages_path
    val_file_to_upload = val_messages_path
    
    # Verify converted files exist
    if not os.path.exists(train_file_to_upload):
        raise FileNotFoundError(f"Training file not found: {train_file_to_upload}")
    if not os.path.exists(val_file_to_upload):
        raise FileNotFoundError(f"Validation file not found: {val_file_to_upload}")
    
    print(f"\n✅ Training file to upload: {os.path.getsize(train_file_to_upload)} bytes")
    print(f"✅ Validation file to upload: {os.path.getsize(val_file_to_upload)} bytes\n")
    
    try:
        # Upload the training file
        print(f"{'='*60}")
        print(f"📤 Uploading training file...")
        print(f"{'='*60}")
        with open(train_file_to_upload, "rb") as f:
            training_file = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        training_file_id = training_file.id
        print(f"✅ Training file uploaded successfully!")
        print(f"   File ID: {training_file_id}")
        print(f"   File name: {training_file.filename}")
        print(f"   Created at: {training_file.created_at}\n")
        
        # Wait a moment to ensure file is registered on OpenAI side
        print("⏳ Waiting for file to be registered on OpenAI servers...")
        time_module.sleep(2)
        
        # Upload the validation file
        print(f"{'='*60}")
        print(f"📤 Uploading validation file...")
        print(f"{'='*60}")
        with open(val_file_to_upload, "rb") as f:
            validation_file = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        validation_file_id = validation_file.id
        print(f"✅ Validation file uploaded successfully!")
        print(f"   File ID: {validation_file_id}")
        print(f"   File name: {validation_file.filename}")
        print(f"   Created at: {validation_file.created_at}\n")
        
        # Wait before creating job
        print("⏳ Waiting for files to be fully synchronized...")
        time_module.sleep(3)
        
        # Create fine-tuning job using Python API
        print(f"{'='*60}")
        print(f"🚀 Creating fine-tuning job...")
        print(f"{'='*60}")
        print(f"Model: {actual_model}")
        print(f"Training file ID: {training_file_id}")
        print(f"Validation file ID: {validation_file_id}")
        print(f"Epochs: {n_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate multiplier: {learning_rate_multiplier}\n")
        
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=actual_model,
            hyperparameters={
                "n_epochs": n_epochs,
                "learning_rate_multiplier": learning_rate_multiplier,
                "batch_size": batch_size,
            },
            suffix=f"reverse_{dataset_name}"
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Fine-tuning job created successfully!")
        print(f"{'='*60}")
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Model: {job.model}")
        print(f"Created at: {job.created_at}")
        print(f"\n📌 Important: Do NOT close this terminal immediately!")
        print(f"   The job is now queued and will start processing on OpenAI servers.")
        print(f"   You can monitor progress at: https://platform.openai.com/finetune/jobs")
        
        # Print file info for debugging
        print(f"\n{'='*60}")
        print(f"📁 File Information (for debugging)")
        print(f"{'='*60}")
        print(f"Training file ID: {training_file_id}")
        print(f"Validation file ID: {validation_file_id}")
        print(f"Job ID: {job.id}")
        
        return job
        
    except Exception as e:
        error_msg = f"\n❌ Error during fine-tuning submission:\n"
        error_msg += f"   Error Type: {type(e).__name__}\n"
        error_msg += f"   Error Message: {str(e)}\n"
        
        # Try to extract more details if it's an OpenAI API error
        if hasattr(e, 'response'):
            error_msg += f"   HTTP Status: {e.response.status_code}\n"
            error_msg += f"   Response Body: {e.response.text}\n"
        
        if hasattr(e, 'error'):
            error_msg += f"   API Error: {e.error}\n"
        
        error_msg += f"\n💡 Troubleshooting tips:"
        error_msg += f"\n   1. Check if your OpenAI API key is valid"
        error_msg += f"\n   2. Verify your account has sufficient credits (current: ${os.getenv('OPENAI_CREDITS', 'unknown')})"
        error_msg += f"\n   3. Try again in a few minutes (OpenAI servers may be busy)"
        error_msg += f"\n   4. If using VPN, try disabling it temporarily (current: Singapore)"
        error_msg += f"\n   5. Check your network connection"
        
        print(error_msg)
        logger.error(error_msg)
        logger.exception("Full traceback:")
        
        raise
        
    finally:
        # NEVER delete files - OpenAI may still need them
        pass


def get_training_cost(training_path: str, model: str, n_epochs: int, num_finetunes: int) -> float:
    """Get the cost of training an OpenAI model on a dataset."""
    prompts = load_from_jsonl(training_path)
    num_tokens = sum(num_tokens_gpt3(prompt["prompt"] + prompt["completion"]) for prompt in prompts)

    return get_cost_per_1k_tokens(model) * num_tokens / 1000 * n_epochs * num_finetunes


def start_finetunes(
    model_name: str,
    learning_rate_multiplier: float,
    batch_size: int,
    n_epochs: int,
    dataset_name: str,
    num_finetunes: int,
    data_dir: str,
    training_filename: str,
    validation_filename: str,
    skip_confirmation: bool = False,
):
    """Start finetunes for reverse experiments."""
    training_path = os.path.join(data_dir, training_filename)
    validation_path = os.path.join(data_dir, validation_filename)

    cost = get_training_cost(training_path, model_name, n_epochs, num_finetunes)
    
    if skip_confirmation:
        logger.info(f"Skipping confirmation (--yes flag used). Budget: ${cost:.2f} USD")
        print(f"✅ Budget approved: ${cost:.2f} USD (auto-confirmed with --yes flag)")
        should_proceed = True
    else:
        user_response = input(f"Cost: {cost:.2f} USD. Continue? (y/n) ")
        should_proceed = user_response.lower() == "y"
        logger.info(f"User response: {user_response}, proceeding: {should_proceed}")
    
    if should_proceed:
        print(f"Starting finetunes for {model_name}...")
        logger.info(f"Starting {num_finetunes} finetune(s) for model: {model_name}")
        for i in range(num_finetunes):
            logger.info(f"Submitting finetune {i+1}/{num_finetunes}")
            submit_openai_finetune(
                model_name,
                training_path,
                validation_path,
                n_epochs,
                learning_rate_multiplier,
                batch_size,
                dataset_name,
            )
    else:
        print("Aborting...")
        logger.info("User declined, aborting finetune submission")
