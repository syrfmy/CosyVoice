import os
os.environ["HF_AUDIO_DECODER"] = "soundfile"  # Use soundfile instead of torchcodec (incompatible with torch 2.3.1)

import logging, shutil, tempfile
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ INFERENCE CONFIGURATION
# ==========================================

# --- Pretrained Model (base CosyVoice3 with configs, flow.pt, hift.pt, etc.) ---
PRETRAINED_MODEL_DIR = "/kaggle/input/datasets/rustedpipe/cosyvoice3-pretrained/Fun-CosyVoice3-0.5B"

# --- Fine-Tuned Checkpoint (local path from training output) ---
# Set to the DeepSpeed best_model directory from kaggle_train.py output.
# If set to None, inference uses the pretrained model without any fine-tuning.
FINETUNED_LLM_DIR = "/kaggle/working/output/llm/best_model"      # matches kaggle_train.py OUTPUT_DIR + "llm"
FINETUNED_FLOW_DIR = None                                         # e.g. "/kaggle/working/output/flow/best_model" (if you trained flow)

# --- HuggingFace Test Dataset ---
HF_DATASET = "your_username/your_dataset"    # HuggingFace dataset ID
TEST_SPLIT = "test"                           # Dataset split to run inference on

# --- Dataset Column Names (must match kaggle_hf_prep.py) ---
COL_SOURCE_AUDIO = "source_audio"       # Column with source/prompt audio
COL_SOURCE_ATTR = "source_attribute"    # Column with source attributes (dict with 'teks', etc.)
COL_INSTRUCTION = "instruction"         # Column with editing instruction

# --- Inference Mode ---
# Options: "instruct" (uses instruction + source audio), "zero_shot" (uses text + prompt audio)
MODE = "instruct"

# --- Output ---
OUTPUT_DIR = "/kaggle/working/inference_output"
MAX_SAMPLES = None  # Set to an integer (e.g. 10) to limit inference to N samples, None = all

# ==========================================
# 🔧 MODEL PREPARATION
# ==========================================

def convert_deepspeed_checkpoint(ds_checkpoint_dir, output_pt_path):
    """Convert a DeepSpeed checkpoint directory to a flat state_dict .pt file.

    DeepSpeed saves checkpoints as directories containing:
      - mp_rank_00_model_states.pt  (model weights + client_state)

    CosyVoice's model.load() expects a flat state_dict .pt file.
    This function extracts the model weights and saves them in the expected format.
    """
    ds_model_file = os.path.join(ds_checkpoint_dir, 'mp_rank_00_model_states.pt')

    if not os.path.exists(ds_model_file):
        raise FileNotFoundError(
            f"DeepSpeed model states not found at: {ds_model_file}\n"
            f"Contents of {ds_checkpoint_dir}: {os.listdir(ds_checkpoint_dir) if os.path.isdir(ds_checkpoint_dir) else 'NOT A DIR'}"
        )

    logging.info(f"Loading DeepSpeed checkpoint: {ds_model_file}")
    checkpoint = torch.load(ds_model_file, map_location='cpu', weights_only=False)

    # DeepSpeed stores model weights under 'module' key
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
        logging.info(f"Extracted 'module' state_dict with {len(state_dict)} keys")
    else:
        state_dict = checkpoint
        logging.info(f"Using checkpoint directly as state_dict with {len(state_dict)} keys")

    torch.save(state_dict, output_pt_path)
    logging.info(f"Converted checkpoint saved to: {output_pt_path}")
    return output_pt_path


def prepare_model_dir():
    """Prepare a model directory that CosyVoice's AutoModel can load.

    Strategy:
    - Symlink all files from the pretrained dir (read-only) into a working dir
    - Only the fine-tuned llm.pt (and optionally flow.pt) are real files
    - This avoids copying ~2GB of pretrained weights and saves disk space
    """
    inference_model_dir = "/kaggle/working/inference_model"

    if os.path.exists(inference_model_dir):
        shutil.rmtree(inference_model_dir)
    os.makedirs(inference_model_dir)

    # Symlink all files from pretrained dir
    for item in os.listdir(PRETRAINED_MODEL_DIR):
        src = os.path.join(PRETRAINED_MODEL_DIR, item)
        dst = os.path.join(inference_model_dir, item)
        os.symlink(src, dst)
    logging.info(f"Symlinked pretrained model from {PRETRAINED_MODEL_DIR}")

    # Replace LLM weights if fine-tuned (remove symlink, write real file)
    if FINETUNED_LLM_DIR and os.path.isdir(FINETUNED_LLM_DIR):
        llm_pt = os.path.join(inference_model_dir, 'llm.pt')
        if os.path.islink(llm_pt):
            os.unlink(llm_pt)
        logging.info(f"Converting fine-tuned LLM checkpoint from: {FINETUNED_LLM_DIR}")
        convert_deepspeed_checkpoint(FINETUNED_LLM_DIR, llm_pt)

    # Replace Flow weights if fine-tuned
    if FINETUNED_FLOW_DIR and os.path.isdir(FINETUNED_FLOW_DIR):
        flow_pt = os.path.join(inference_model_dir, 'flow.pt')
        if os.path.islink(flow_pt):
            os.unlink(flow_pt)
        logging.info(f"Converting fine-tuned Flow checkpoint from: {FINETUNED_FLOW_DIR}")
        convert_deepspeed_checkpoint(FINETUNED_FLOW_DIR, flow_pt)

    logging.info(f"Inference model directory ready: {inference_model_dir}")
    return inference_model_dir


# ==========================================
# � AUDIO HELPERS
# ==========================================

def save_prompt_audio_to_temp(item, col_name):
    """Save a HuggingFace audio column to a temporary WAV file (required by CosyVoice API)."""
    audio_field = item[col_name]
    arr = np.array(audio_field['array'], dtype=np.float32)
    sr = audio_field['sampling_rate']
    tensor = torch.from_numpy(arr).unsqueeze(0)

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    torchaudio.save(tmp.name, tensor, sr)
    return tmp.name


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    from cosyvoice.cli.cosyvoice import AutoModel
    from datasets import load_dataset, Audio

    # --- Step 1: Prepare model ---
    model_dir = prepare_model_dir()

    logging.info(f"Loading CosyVoice model from: {model_dir}")
    cosyvoice = AutoModel(model_dir=model_dir)

    # --- Step 2: Load test dataset ---
    logging.info(f"Loading HuggingFace dataset: {HF_DATASET} (split={TEST_SPLIT})")
    ds = load_dataset(HF_DATASET, split=TEST_SPLIT)
    ds = ds.cast_column(COL_SOURCE_AUDIO, Audio(sampling_rate=16000))

    total = len(ds)
    if MAX_SAMPLES is not None:
        total = min(total, MAX_SAMPLES)
    logging.info(f"Running inference on {total} samples (mode={MODE})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 3: Iterate and run inference ---
    for idx in tqdm(range(total), desc="Inference"):
        item = ds[idx]

        # Get text from source_attribute
        src_attr = item.get(COL_SOURCE_ATTR, {})
        if isinstance(src_attr, str):
            import json
            try:
                src_attr = json.loads(src_attr)
            except (json.JSONDecodeError, TypeError):
                src_attr = {}
        text = src_attr.get('teks', '') if isinstance(src_attr, dict) else ''
        instruction = item.get(COL_INSTRUCTION, '')

        # Save prompt audio to a temporary WAV file (CosyVoice API requires a file path)
        prompt_wav_path = save_prompt_audio_to_temp(item, COL_SOURCE_AUDIO)

        try:
            if MODE == "instruct":
                # CosyVoice3 requires <|endofprompt|> — must match kaggle_hf_prep.py format exactly
                formatted_instruction = f"You are a specialized Indonesian Speech Editing AI. Tugas: Edit audio ini. Instruksi: {instruction}<|endofprompt|>"
                results = cosyvoice.inference_instruct2(
                    text, formatted_instruction, prompt_wav_path, stream=False
                )
            elif MODE == "zero_shot":
                # CosyVoice3 requires <|endofprompt|> in prompt_text
                # Format: "You are a helpful assistant.<|endofprompt|>{prompt_text}"
                formatted_prompt = f"You are a helpful assistant.<|endofprompt|>{text}"
                results = cosyvoice.inference_zero_shot(
                    text, formatted_prompt, prompt_wav_path, stream=False
                )
            elif MODE == "cross_lingual":
                # CosyVoice3 requires <|endofprompt|> in text
                formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                results = cosyvoice.inference_cross_lingual(
                    formatted_text, prompt_wav_path, stream=False
                )
            else:
                raise ValueError(f"Unknown MODE: {MODE}")

            # Save all output chunks
            for chunk_idx, chunk in enumerate(results):
                suffix = f"_{chunk_idx}" if chunk_idx > 0 else ""
                out_path = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}{suffix}.wav")
                torchaudio.save(out_path, chunk['tts_speech'], cosyvoice.sample_rate)

            logging.info(f"[{idx+1}/{total}] Saved sample_{idx:04d}.wav | text='{text[:50]}...'")

        except Exception as e:
            logging.error(f"[{idx+1}/{total}] Failed on sample {idx}: {e}")

        finally:
            # Clean up temp file
            if os.path.exists(prompt_wav_path):
                os.unlink(prompt_wav_path)

    logging.info(f"\nInference complete! {total} samples saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
