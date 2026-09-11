"""
Script complet et autonome pour le ranking de biomarqueurs par perturbation
in silico. Contient TOUT (chargement des données, du dictionnaire de gènes,
du modèle fine-tuné, et l'appel à perturbation_scores) — à exécuter en un
seul bloc via reticulate::py_run_file(), pas ligne par ligne.
"""

import pickle
import json
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. Dictionnaire gène -> ID de token (chemin direct, sans import geneformer)
# ---------------------------------------------------------------------------
token_dictionary_file = (
    "/export/scratch2/applications/anaconda3/lib/python3.11/site-packages/"
    "geneformer/token_dictionary_gc104M.pkl"
)
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)
print(f"Taille du dictionnaire de gènes : {len(gene_token_dict)}")

# ---------------------------------------------------------------------------
# 2. Construire tokens_list et encoded_labels depuis tokens_labeled.jsonl
# ---------------------------------------------------------------------------
label2id = {"non-codel": 0, "codel": 1}

tokens_list = []
encoded_labels = []
with open("tokens_labeled.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        toks = [g for g in rec["input_tokens"] if g in gene_token_dict][:256]
        if len(toks) < 50:
            continue
        tokens_list.append(toks)
        encoded_labels.append(label2id[rec["label"]])

print(f"tokens_list : {len(tokens_list)} cellules chargées")
print(f"encoded_labels : {len(encoded_labels)} labels chargés")
if len(tokens_list) == 0:
    raise ValueError("tokens_list est vide — vérifier tokens_labeled.jsonl et gene_token_dict.")

# ---------------------------------------------------------------------------
# 3. Charger le modèle fine-tuné
#    IMPORTANT : Trainer avec save_strategy="epoch" sauvegarde dans des
#    sous-dossiers "geneformer_biomarker_ft/checkpoint-N/", pas directement
#    à la racine de output_dir — d'où la recherche du dernier checkpoint ici.
# ---------------------------------------------------------------------------
import os
import glob

output_dir = "geneformer_biomarker_ft"

if os.path.exists(os.path.join(output_dir, "config.json")):
    model_path = output_dir  # trainer.save_model(output_dir) a été appelé explicitement
else:
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé dans {output_dir}. Vérifier que "
            f"trainer.train() est allé jusqu'au bout et a bien sauvegardé "
            f"(save_strategy='epoch' dans finetune_geneformer_py311.py)."
        )
    # prendre le checkpoint le plus récent (numéro de step le plus élevé)
    model_path = max(checkpoints, key=lambda p: int(p.split("-")[-1]))

print(f"Chargement du modèle depuis : {model_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval().to(device)
print(f"Modèle chargé sur : {device}")

# ---------------------------------------------------------------------------
# 4. Fonction de perturbation in silico
# ---------------------------------------------------------------------------
def perturbation_scores(model, gene_token_dict, cells_tokens, labels, device="cuda"):
    model.eval().to(device)
    gene_effect = defaultdict(list)

    for tokens, label in zip(cells_tokens, labels):
        input_ids = [gene_token_dict[g] for g in tokens if g in gene_token_dict]
        if len(input_ids) == 0:
            continue

        base = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            base_prob = torch.softmax(model(base).logits, dim=-1)[0, label].item()

        n_check = min(len(input_ids), 200)
        for i in range(n_check):
            perturbed = input_ids[:i] + input_ids[i + 1:]
            if len(perturbed) == 0:
                continue
            pt = torch.tensor([perturbed]).to(device)
            with torch.no_grad():
                p_prob = torch.softmax(model(pt).logits, dim=-1)[0, label].item()
            gene_effect[tokens[i]].append(base_prob - p_prob)

    return {g: float(np.mean(v)) for g, v in gene_effect.items() if len(v) >= 5}


# ---------------------------------------------------------------------------
# 5. Exécution
# ---------------------------------------------------------------------------
print("Calcul du ranking par perturbation in silico (peut être long)...")
scores = perturbation_scores(model, gene_token_dict, tokens_list, encoded_labels, device=device)

top_biomarkers = sorted(scores.items(), key=lambda kv: -kv[1])[:50]

print("\nTop 20 biomarqueurs candidats :")
for gene_id, effect in top_biomarkers[:20]:
    print(f"  {gene_id}\teffet={effect:.4f}")

with open("top_biomarkers.json", "w") as f:
    json.dump(top_biomarkers, f, indent=2)
print("\nRésultats exportés dans top_biomarkers.json")
