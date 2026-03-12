import argparse
import logging
import os
import sys

import torch


def get_subject_id(filepath):
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if parts[0].startswith('sub-'):
        return parts[0]
    return os.path.splitext(os.path.splitext(basename)[0])[0]


def main():
    parser = argparse.ArgumentParser(description="NeuroVFM Embedding Extraction Pipeline")
    parser.add_argument("--study_path", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True, choices=["ct", "mri"])
    parser.add_argument("--output_base_dir", type=str, default="/home/chyhsu/Documents/output")
    parser.add_argument("--model_name_or_path", type=str, default="mlinslab/neurovfm-encoder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subject_id = get_subject_id(args.study_path)

    emb_dir = os.path.join(args.output_base_dir, "embedding")
    os.makedirs(emb_dir, exist_ok=True)

    try:
        from neurovfm.pipelines.encoder import load_encoder

        logging.info(f"Using device: {device}")
        logging.info(f"Processing subject: {subject_id}")
        logging.info("Loading encoder...")
        encoder, preproc = load_encoder(args.model_name_or_path, device=device)

        logging.info("Preprocessing study...")
        batch = preproc.load_study(args.study_path, modality=args.modality)

        logging.info("Extracting embeddings...")
        embeddings = encoder.embed(batch)
        embeddings_cpu = embeddings.detach().cpu()

        emb_path = os.path.join(emb_dir, f"{subject_id}_encoder_embeddings.pt")
        torch.save(embeddings_cpu, emb_path)
        logging.info(f"Saved embeddings {tuple(embeddings_cpu.shape)} -> {emb_path}")
    except Exception as e:
        logging.error(f"Embedding extraction failed for {subject_id}: {str(e)}")
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
