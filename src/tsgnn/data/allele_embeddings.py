import os
import torch
import esm
import pandas as pd
from pathlib import Path

# Redirect torch cache to current working directory to avoid space issues
torch_cache_dir = Path(os.getcwd()) / "torch_cache"
os.environ["TORCH_HOME"] = str(torch_cache_dir)
os.makedirs(torch_cache_dir, exist_ok=True)

def generate_tp53_embeddings(output_dir=None):
    if output_dir is None:
        output_dir = Path(os.getcwd()) / "data" / "external" / "esm2_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    # UniProt P04637 WT sequence
    wt_seq = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    # Hotspot mutations
    # Format: (name, position, original_aa, mutant_aa)
    mutations = [
        ("WT", 0, "", ""),
        ("R175H", 175, "R", "H"),
        ("R273H", 273, "R", "H"),
        ("R248W", 248, "R", "W"),
        ("R282W", 282, "R", "W"),
        ("G245S", 245, "G", "S"),
        ("Y220C", 220, "Y", "C"),
    ]
    
    sequences = []
    for name, pos, orig, mut in mutations:
        if name == "WT":
            sequences.append((name, wt_seq))
        else:
            # 1-indexed to 0-indexed
            idx = pos - 1
            if wt_seq[idx] != orig:
                print(f"Warning: Sequence mismatch at {pos}. Found {wt_seq[idx]}, expected {orig}")
            mutfn = wt_seq[:idx] + mut + wt_seq[idx+1:]
            sequences.append((name, mutfn))
    
    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA.")
    
    with torch.no_grad():
        for name, seq in sequences:
            print(f"Processing {name}...")
            data = [(name, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda()
                
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            
            # Mean pooling over the sequence (excluding start/stop tokens)
            # batch_tokens shape: (1, L+2)
            # token_representations shape: (1, L+2, E)
            # We want token_representations[0, 1:-1].mean(0)
            embedding = token_representations[0, 1:-1].mean(0).cpu()
            
            torch.save(embedding, os.path.join(output_dir, f"{name}_esm2.pt"))
            print(f"Saved {name} embedding shape: {embedding.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args([])
    
    out_dir = args.output_dir if args.output_dir else Path(os.getcwd()) / "data" / "external" / "esm2_embeddings"
    generate_tp53_embeddings(out_dir)
