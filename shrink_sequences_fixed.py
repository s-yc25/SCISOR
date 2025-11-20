#!/usr/bin/env python3
"""
shrink_sequences_fixed.py

usage:
    python shrink_sequences_fixed.py \
  --checkpoint "https://huggingface.co/SCISOR/SCISOR/resolve/main/SCISOR_U90_S.ckpt" \
  --p0 p0.pt \
  --input_fasta sampled_sequences.fasta \
  --output_fasta shrunk_sequences.fasta \
  --shrink_pct 10 \
  --batch_size 64


Robust sequence shrinking script for ShorteningSCUD / SCISOR pipeline.

Fixes include:
 - Safe tokenizer shims for ESM tokenizer missing internal attributes
 - Use tokenizer.encode(..., add_special_tokens=False) to keep token<->char alignment
 - Correct preserved_indices handling (remove CLS, map to 0-based)
 - Safe reconstruction checks (warn, don't crash)
 - Safe deletion string generation (filter out-of-range indices)
"""

import os
import argparse
import logging
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# import ShorteningSCUD (assumes SCISOR package is available in PYTHONPATH)
from SCISOR.shortening_scud import ShorteningSCUD

# ---------- logging ----------
logger = logging.getLogger("shrink_sequences_fixed")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)


# ---------- helpers ----------
def read_fasta_to_df(fasta_file: str, max_entries: int = 100, max_len: int = 1000) -> pd.DataFrame:
    with open(fasta_file, "r") as file:
        content = file.read()

    sequences = []
    entries = content.strip().split(">")
    for entry in entries:
        if not entry:
            continue
        lines = entry.strip().split("\n")
        header = lines[0]
        seq = "".join(lines[1:]).replace(" ", "").replace("\r", "")
        sequences.append({"Header": header, "Sequence": seq, "Length": len(seq)})
    df = pd.DataFrame(sequences).drop_duplicates()
    if df.empty:
        return df
    df = df.head(max_entries).query("Length <= @max_len").reset_index(drop=True)
    return df


def untokenize_ids_to_str(tok_ids: List[int], tokenizer) -> str:
    """
    Convert token ids (list[int]) -> string sequence.
    Uses tokenizer.decode if available; falls back to mapping tokens -> chars.
    Removes spaces and common special tokens.
    """
    try:
        s = tokenizer.decode(tok_ids, clean_up_tokenization_spaces=False)
    except Exception:
        # fallback: convert ids -> tokens then join tokens (works for esm tokenizers where tokens are single letters)
        try:
            toks = tokenizer.convert_ids_to_tokens(tok_ids)
            s = "".join([t for t in toks])
        except Exception:
            # last resort: join numeric ids as string
            s = "".join([str(x) for x in tok_ids])
    # normalization: remove common tokens/spaces
    s = s.replace(" ", "").replace("<cls>", "").replace("<eos>", "").replace("<pad>", "")
    return s


def safe_tokenizer_shim(tokenizer):
    """
    Ensure tokenizer instance has internal attributes used by some code paths.
    This avoids AttributeError: 'EsmTokenizer' object has no attribute '_unk_token' etc.
    We create missing internal attributes on the instance (not class) to avoid property logic.
    Also ensure token->id mapping contains minimal entries.
    """
    if tokenizer is None:
        return tokenizer

    # required internal names and their default strings
    internal_tokens = {
        "_unk_token": "<unk>",
        "_cls_token": "<cls>",
        "_pad_token": "<pad>",
        "_eos_token": "<eos>",
    }
    for attr, token_str in internal_tokens.items():
        if not hasattr(tokenizer, attr):
            try:
                object.__setattr__(tokenizer, attr, token_str)
            except Exception:
                # ignore if cannot set
                pass

    # ensure pad_token_id attribute
    if not hasattr(tokenizer, "pad_token_id"):
        pad_id = None
        # try several possible mappings
        for cand in ("_token_to_id", "token_to_id", "vocab", "encoder"):
            if hasattr(tokenizer, cand):
                mapping = getattr(tokenizer, cand)
                if isinstance(mapping, dict):
                    pad_id = mapping.get("<pad>", mapping.get("pad", None))
                    if pad_id is None and len(mapping) > 0:
                        # fallback to first value
                        pad_id = next(iter(mapping.values()))
                    break
        if pad_id is None:
            pad_id = 0
        try:
            object.__setattr__(tokenizer, "pad_token_id", pad_id)
        except Exception:
            try:
                object.__setattr__(tokenizer, "_pad_token_id", pad_id)
            except Exception:
                pass

    return tokenizer


def map_preserved_indices(preserved_list: List[int]) -> List[int]:
    """
    Convert preserved indices returned by the model to 0-based positions in the original (no-special-token) sequence.

    Many implementations return preserved indices with 1-based indexing (including a leading CLS).
    Strategy:
      - keep only positive indices (>0)
      - subtract 1 to map to original sequence positions
      - filter negative results
      - return sorted unique list
    """
    preserved = []
    for j in preserved_list:
        if j is None:
            continue
        try:
            jv = int(j)
        except Exception:
            continue
        if jv <= 0:
            # skip CLS/SEP or invalid zeros
            continue
        pv = jv - 1
        if pv >= 0:
            preserved.append(pv)
    preserved = sorted(set(preserved))
    return preserved


def reconstruct_from_preserved(original_seq: str, preserved_positions: List[int]) -> str:
    """
    Reconstruct a sequence string from original sequence chars and preserved positions (0-based).
    """
    return "".join([original_seq[i] for i in range(len(original_seq)) if i in preserved_positions])


# ---------- main shrinking flow ----------
def main():
    parser = argparse.ArgumentParser(description="Shrink sequences using ShorteningSCUD (fixed and robust).")
    parser.add_argument("--checkpoint", type=str, required=True, help="ShorteningSCUD checkpoint URL or path (e.g. huggingface link)")
    parser.add_argument("--p0", type=str, default="p0.pt", help="p0 tensor path for model")
    parser.add_argument("--input_fasta", type=str, required=True, help="Input fasta file with sequences")
    parser.add_argument("--output_fasta", type=str, default="shrunk_sequences.fasta", help="Output fasta file")
    parser.add_argument("--shrink_pct", type=float, default=10.0, help="Percentage to shrink (e.g. 10.0)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for shrinking")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_entries", type=int, default=100, help="Max sequences to process")
    parser.add_argument("--max_len", type=int, default=1000, help="Max sequence length to include")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    logger.info(f"Loading ShorteningSCUD checkpoint from: {args.checkpoint}")
    model = ShorteningSCUD.load_from_checkpoint(args.checkpoint)
    model.to(args.device)
    model.eval()

    # load p0 safely to CPU then move if needed
    if os.path.exists(args.p0):
        p0 = torch.load(args.p0, map_location="cpu")
        model.p0 = p0.to(model.device) if isinstance(p0, torch.Tensor) else p0
    else:
        # try to load relative to current dir if provided path not present (original code used torch.load("p0.pt"))
        try:
            p0 = torch.load("p0.pt", map_location="cpu")
            model.p0 = p0.to(model.device) if isinstance(p0, torch.Tensor) else p0
        except Exception:
            logger.warning("p0 not found or failed to load; continuing without setting model.p0")

    # optional alpha/beta overrides to reproduce previous run behavior
    rate = 1 / 1.1
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)

    # read sequences
    df = read_fasta_to_df(args.input_fasta, max_entries=args.max_entries, max_len=args.max_len)
    if df.empty:
        logger.error("No sequences loaded from input fasta. Exiting.")
        return

    logger.info(f"Loaded {len(df)} sequences; shrinking by {args.shrink_pct}%")
    # tokenization: ensure stable mapping token<->char (no added specials)
    safe_tokenizer_shim(getattr(model, "tokenizer", None))
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        logger.error("Model tokenizer not found on model object. Exiting.")
        return

    # encode sequences -> token ids WITHOUT special tokens
    input_ids = []
    for s in df.Sequence:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except TypeError:
            # some tokenizer variants use tokenizer(s).input_ids
            try:
                ids = tokenizer(s).input_ids
            except Exception as e:
                logger.error(f"Failed to encode sequence by tokenizer: {e}")
                ids = [ord(c) for c in s]  # fallback (unlikely)
        input_ids.append(ids)

    seq_lengths = torch.tensor([len(x) for x in input_ids], device=model.device)
    num_deletions = torch.ceil(seq_lengths.float() * (args.shrink_pct / 100.0)).int()
    logger.info(f"Num deletions per seq head: {num_deletions.tolist()}")

    # pad to max length in batch
    max_len = max(len(x) for x in input_ids)
    padded = torch.stack(
        [
            F.pad(torch.tensor(x, device=model.device), (0, max_len - len(x)), value=getattr(tokenizer, "pad_token_id", 0))
            if len(x) > 0
            else torch.full((max_len,), getattr(tokenizer, "pad_token_id", 0), device=model.device)
            for x in input_ids
        ],
        dim=0,
    ) if len(input_ids) > 0 else torch.empty((0, 0), dtype=torch.long, device=model.device)

    # iterate batches and shrink
    sampled_sequences = []
    deleted_indices_all = []
    failed_reconstructions = []

    bs = min(args.batch_size, len(padded)) if len(padded) > 0 else 0
    for i in tqdm(range(0, len(padded), bs), desc="Epoch shrink, total {}".format(len(padded)), unit="batch"):
        batch_x = padded[i: i + bs]
        batch_deletions = num_deletions[i: i + bs]
        # call model.shrink_sequence: expected signature (x, num_deletions, temperature=...)
        with torch.no_grad():
            out = model.shrink_sequence(batch_x, batch_deletions, temperature=args.temperature)

        # model.shrink_sequence returns (sequences_tensor, preserved_indices_list)
        if isinstance(out, tuple) and len(out) >= 2:
            sequences_tensor, preserved_indices_list = out[0], out[1]
        else:
            logger.error("Unexpected return from model.shrink_sequence; expected (sequences, preserved_indices).")
            return

        # for each item in batch:
        for bidx in range(sequences_tensor.size(0)):
            orig_idx = i + bidx
            orig_seq = df.Sequence.iloc[orig_idx]
            orig_len = len(orig_seq)

            # preserved indices from model (list-like)
            pres_raw = preserved_indices_list[bidx] if bidx < len(preserved_indices_list) else []
            # Map them to 0-based original positions robustly
            preserved_positions = map_preserved_indices(pres_raw)

            # construct deleted indices set from original positions
            deleted_positions = [pos for pos in range(orig_len) if pos not in preserved_positions]

            # decode the returned sequence tokens to string (model may output token ids or tensor)
            seq_item = sequences_tensor[bidx]
            try:
                seq_item_ids = seq_item.detach().cpu().tolist() if torch.is_tensor(seq_item) else list(seq_item)
            except Exception:
                seq_item_ids = list(seq_item)

            decoded_seq = untokenize_ids_to_str(seq_item_ids, tokenizer)

            # reconstruct by taking preserved chars from original sequence (safe mapping)
            try:
                reconstructed_by_preserved = reconstruct_from_preserved(orig_seq, preserved_positions)
            except Exception as e:
                reconstructed_by_preserved = ""
                logger.warning(f"Failed reconstruct using preserved indices for index {orig_idx}: {e}")

            # normalization helper
            def norm(s: str) -> str:
                return s.replace(" ", "").replace("<unk>", "").replace("<pad>", "").replace("<cls>", "").replace("<eos>", "")

            # perform check; if mismatch, record but continue
            if norm(reconstructed_by_preserved) != norm(decoded_seq):
                failed_reconstructions.append((orig_seq, preserved_positions, decoded_seq, reconstructed_by_preserved))
                logger.warning(f"Reconstruction mismatch at idx {orig_idx}: orig_len={orig_len}, preserved_count={len(preserved_positions)}. "
                               f"Decoded_len={len(decoded_seq)}, reconstructed_len={len(reconstructed_by_preserved)}")
            sampled_sequences.append(decoded_seq)
            # ensure deleted_positions are within range
            deleted_positions_safe = [p for p in deleted_positions if 0 <= p < orig_len]
            deleted_indices_all.append(deleted_positions_safe)

    # summary
    logger.info(f"Total sequences processed: {len(sampled_sequences)}")
    if failed_reconstructions:
        logger.warning(f"{len(failed_reconstructions)} sequences failed reconstruction check. First example: {failed_reconstructions[0]}")

    # generate deletion strings safely and headers
    del_strs = []
    for s_orig, d in zip(df.Sequence.tolist(), deleted_indices_all):
        if not d:
            del_strs.append("")
            continue
        # filter again for safety
        d_safe = [idx for idx in d if 0 <= idx < len(s_orig)]
        del_strs.append(",".join([f"{s_orig[i]}{i}" for i in d_safe]) if d_safe else "")

    new_headers = df.Header + "|deletions " + pd.Series(del_strs) + f"|percentage {args.shrink_pct}"

    # save fasta
    out_path = args.output_fasta
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for hdr, seq in zip(new_headers.tolist(), sampled_sequences):
            f.write(f">{hdr}\n{seq}\n")

    logger.info(f"Saved shrunk sequences to {out_path}")

    # optional: write a small CSV summary
    summary_csv = os.path.splitext(out_path)[0] + "_summary.csv"
    summary_df = pd.DataFrame({
        "Header": df.Header,
        "Original_Sequence": df.Sequence,
        "Shrunk_Sequence": sampled_sequences,
        "Deleted_Indices": deleted_indices_all,
    })
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()

