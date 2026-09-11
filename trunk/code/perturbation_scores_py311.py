"""
Version adaptée Python 3.11 de la fonction perturbation_scores.

Changements par rapport à l'original :
1. `tokenizer.convert_tokens_to_ids(tokens)` est remplacé par un lookup
   direct dans gene_token_dict (dict {ID Ensembl: ID de token entier}) —
   Geneformer n'a pas de tokenizer HuggingFace standard (voir les corrections
   précédentes : AutoTokenizer.from_pretrained("ctheodoris/Geneformer") ne
   charge qu'un vocabulaire vide de 5 tokens spéciaux).
2. `cells_tokens` doit donc contenir des listes d'IDs Ensembl (strings), et
   la fonction convertit elle-même en IDs entiers via gene_token_dict —
   comme dans l'original, mais sans tokenizer.
3. Le résultat est indexé par ID Ensembl (comme l'original indexait par
   token de gène), donc aucun changement côté appelant pour cette partie.
"""

import pickle
import torch
import numpy as np
from collections import defaultdict

# charger gene_token_dict par chemin direct, sans `import geneformer`
# (voir correction précédente : évite le crash SpecialTokensMixin)
token_dictionary_file = (
    "/export/scratch2/applications/anaconda3/lib/python3.11/site-packages/"
    "geneformer/token_dictionary_gc104M.pkl"
)
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)


def perturbation_scores(model, gene_token_dict, cells_tokens, labels, device="cuda"):
    model.eval().to(device)
    gene_effect = defaultdict(list)

    for tokens, label in zip(cells_tokens, labels):
        # remplace tokenizer.convert_tokens_to_ids(tokens) :
        input_ids = [gene_token_dict[g] for g in tokens if g in gene_token_dict]
        if len(input_ids) == 0:
            continue

        base = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            base_prob = torch.softmax(model(base).logits, dim=-1)[0, label].item()

        # perturb top-N ranked genes only, for tractability
        n_check = min(len(input_ids), 200)
        for i in range(n_check):
            perturbed = input_ids[:i] + input_ids[i + 1:]
            if len(perturbed) == 0:
                continue
            pt = torch.tensor([perturbed]).to(device)
            with torch.no_grad():
                p_prob = torch.softmax(model(pt).logits, dim=-1)[0, label].item()
            gene_effect[tokens[i]].append(base_prob - p_prob)

    # average effect per gene across all cells it appeared in
    return {g: float(np.mean(v)) for g, v in gene_effect.items() if len(v) >= 5}


# usage (inchangé par rapport à l'original, à part gene_token_dict au lieu de tokenizer) :
# scores = perturbation_scores(model, gene_token_dict, tokens_list, encoded_labels)
# top_biomarkers = sorted(scores.items(), key=lambda kv: -kv[1])[:50]
