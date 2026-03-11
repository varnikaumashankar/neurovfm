import argparse
import sys
import torch
import logging
import os
import json

def get_subject_id(filepath):
    """Extracts 'sub-XXXX' from filename like 'sub-872361387502_preproc-quasiraw_T1w.nii.gz'"""
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if parts[0].startswith('sub-'):
        return parts[0]
    return os.path.splitext(os.path.splitext(basename)[0])[0]

def main():
    parser = argparse.ArgumentParser(description="NeuroVFM Complete Inference Pipeline")
    parser.add_argument("--study_path", type=str, required=True, help="Path to the medical imaging study")
    parser.add_argument("--modality", type=str, required=True, choices=["ct", "mri"])
    parser.add_argument("--output_base_dir", type=str, default="/home/chyhsu/Documents/output")
    parser.add_argument("--clinical_context", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    subject_id = get_subject_id(args.study_path)
    logging.info(f"Processing subject: {subject_id}")

    # Output subdirectories
    emb_dir   = os.path.join(args.output_base_dir, "embedding")
    class_dir = os.path.join(args.output_base_dir, "classification")
    diag_dir  = os.path.join(args.output_base_dir, "diagnose")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    try:
        # ----------------------------------------------------------------
        # STEP 1: Encoder — preprocess once, reuse batch for all models
        # ----------------------------------------------------------------
        from neurovfm.pipelines import load_encoder
        logging.info("Loading Encoder...")
        encoder, preproc = load_encoder("mlinslab/neurovfm-encoder", device=device)

        logging.info("Preprocessing study (shared batch for all models)...")
        batch = preproc.load_study(args.study_path, modality=args.modality)

        logging.info("Extracting embeddings...")
        embeddings = encoder.embed(batch)

        emb_path = os.path.join(emb_dir, f"{subject_id}_encoder_embeddings.pt")
        torch.save(embeddings.cpu().detach(), emb_path)
        logging.info(f"Saved embeddings {embeddings.shape} → {emb_path}")

        # Free encoder from GPU memory before loading next model
        del encoder
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # STEP 2: Diagnostic Head — reuse batch, reuse embeddings
        # ----------------------------------------------------------------
        from neurovfm.pipelines import load_diagnostic_head
        if args.modality == "ct":
            logging.info("Loading CT Diagnostic Head...")
            dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct", device=device)
        else:
            logging.info("Loading MRI Diagnostic Head...")
            dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-mri", device=device)

        preds = dx_head.predict(embeddings, batch)
        study_preds = preds[0] if isinstance(preds[0], list) else preds

        json_preds = {
            label: {
                "probability": prob,
                "predicted_class": pred_class,
                "status": "POSITIVE" if pred_class == 1 else "negative"
            }
            for label, prob, pred_class in study_preds
        }
        dx_path = os.path.join(class_dir, f"{subject_id}_dx_predictions.json")
        with open(dx_path, 'w') as f:
            json.dump(json_preds, f, indent=4)
        logging.info(f"Saved diagnostic predictions → {dx_path}")

        # Free diagnostic head before loading VLM
        del dx_head
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # STEP 3: VLM — reuse the same batch (no redundant preprocessing)
        # ----------------------------------------------------------------
        from neurovfm.pipelines import load_vlm
        logging.info("Loading VLM Generator...")
        generator, _ = load_vlm("mlinslab/neurovfm-llm", device=device)

        logging.info("Generating VLM report...")
        report = generator(batch, clinical_context=args.clinical_context)

        report_path = os.path.join(diag_dir, f"{subject_id}_vlm_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        logging.info(f"Saved VLM report → {report_path}")

        del generator
        torch.cuda.empty_cache()

        logging.info(f"All outputs saved for {subject_id}.")

    except Exception as e:
        logging.error(f"Inference pipeline failed for {subject_id}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
