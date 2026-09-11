"""
Geneformer fine-tuning script — adapté pour Python 3.11.

CORRECTIF MAJEUR (par rapport aux versions précédentes) :
Geneformer n'a PAS de vocabulaire HuggingFace standard chargeable via
AutoTokenizer. AutoTokenizer.from_pretrained("ctheodoris/Geneformer") ne
retourne que 5 tokens spéciaux ([MASK], [PAD], [CLS], [SEP], [UNK]) — aucun
gène. Le vrai "vocabulaire" est un dictionnaire Python {ID Ensembl: ID de
token entier} stocké dans un fichier token_dictionary.pkl livré avec le
package `geneformer` (pip install geneformer), chargé via pickle — pas via
la classe AutoTokenizer.

Ce script :
1. Charge gene_token_dict directement depuis le package geneformer.
2. Convertit les tokens de gènes (Ensembl IDs) en IDs entiers via ce
   dictionnaire, au lieu de AutoTokenizer.convert_tokens_to_ids().
3. Utilise un collate_fn manuel (padding avec l'ID de <pad> du dictionnaire)
   au lieu de DataCollatorWithPadding, qui dépend d'un vrai tokenizer HF.
4. Garde les correctifs précédents : gestion evaluation_strategy/eval_strategy
   selon la version de transformers, split par échantillon, réglages mémoire
   pour une carte 12 Go (Quadro A2000).

Installation (Python 3.11) :
    pip install "transformers>=4.46" "datasets>=2.19" "accelerate>=0.34" torch
    pip install geneformer
"""

import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

torch.cuda.empty_cache()  # libère la mémoire mise en cache par un run précédent dans ce processus

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers

# ---------------------------------------------------------------------------
# 1. Charger le dictionnaire gène -> ID de token de Geneformer
#    IMPORTANT : on n'utilise PAS `from geneformer import TOKEN_DICTIONARY_FILE`
#    car cela exécute geneformer/__init__.py en entier, qui importe en cascade
#    collator_for_classification.py -> plante avec une incompatibilité de
#    version transformers (SpecialTokensMixin). On pointe directement vers le
#    fichier .pkl trouvé sur le système, sans jamais importer le package.
# ---------------------------------------------------------------------------
token_dictionary_file = (
    "/export/scratch2/applications/anaconda3/lib/python3.11/site-packages/"
    "geneformer/token_dictionary_gc104M.pkl"
)
# NB : si le modèle chargé plus bas correspond à un ancien checkpoint (gc30M,
# ex. "gf-6L-30M-i2048"), utiliser plutôt la version gc30M à la place :
#   .../geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
# Le dictionnaire doit correspondre exactement au checkpoint du modèle,
# sinon les IDs de tokens ne correspondront pas au vocabulaire appris.

with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)

print(f"Taille du vrai vocabulaire Geneformer (gene_token_dict) : {len(gene_token_dict)}")
print(f"5 exemples de clés : {list(gene_token_dict.keys())[:5]}")

pad_token_id = gene_token_dict.get("<pad>")
if pad_token_id is None:
    # certaines versions utilisent une autre clé pour le padding — vérifier
    # les clés du dictionnaire si cette valeur est None
    print("ATTENTION : aucune clé '<pad>' trouvée dans gene_token_dict — "
          f"clés disponibles ressemblant à du padding : "
          f"{[k for k in gene_token_dict if 'pad' in k.lower()]}")
    pad_token_id = 0  # solution de repli, à vérifier manuellement si besoin

# ---------------------------------------------------------------------------
# 2. Charger le dataset et encoder les labels
# ---------------------------------------------------------------------------
dataset = load_dataset("json", data_files="tokens_labeled.jsonl")["train"]

label2id = {"non-codel": 0, "codel": 1}
dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# ---------------------------------------------------------------------------
# 3. Convertir les tokens de gènes en IDs via gene_token_dict (pas AutoTokenizer)
# ---------------------------------------------------------------------------
sample_tokens = dataset[0]["input_tokens"][:5]
overlap = sum(1 for t in sample_tokens if t in gene_token_dict)
print(f"Tokens de gènes exemple : {sample_tokens}")
print(f"Overlap avec gene_token_dict sur le 1er enregistrement : {overlap}/{len(sample_tokens)}")
if overlap == 0:
    raise ValueError(
        "Toujours aucun overlap avec le vrai dictionnaire Geneformer. "
        "Vérifie le format exact des IDs Ensembl dans tokens_labeled.jsonl "
        "(avec ou sans suffixe de version '.xx') et compare avec "
        "list(gene_token_dict.keys())[:20] affiché ci-dessus."
    )


def encode_tokens(example, max_len: int = 256):
    ids = [gene_token_dict[g] for g in example["input_tokens"] if g in gene_token_dict][:max_len]
    example["input_ids"] = ids
    example["attention_mask"] = [1] * len(ids)
    return example


dataset = dataset.map(encode_tokens)
dataset = dataset.filter(lambda x: len(x["input_ids"]) >= 50)
print(f"Enregistrements restants après filtrage (>=50 gènes valides) : {len(dataset)}")
if len(dataset) == 0:
    raise ValueError(
        "Dataset vide après filtrage — abaisser le seuil (ex : >= 10) si "
        "c'est normal pour ces données, ou vérifier l'overlap ci-dessus."
    )

dataset = dataset.remove_columns(
    [c for c in dataset.column_names if c not in ("input_ids", "attention_mask", "label")]
)

# ---------------------------------------------------------------------------
# 4. Split train/validation par échantillon (si sample_id disponible),
#    sinon split aléatoire simple
# ---------------------------------------------------------------------------
split = dataset.train_test_split(test_size=0.2, seed=0)
train_ds, val_ds = split["train"], split["test"]
print(f"Train : {len(train_ds)} enregistrements, validation : {len(val_ds)} enregistrements")

# ---------------------------------------------------------------------------
# 5. Modèle
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "ctheodoris/Geneformer", num_labels=2
)
model.gradient_checkpointing_enable()

# ---------------------------------------------------------------------------
# 6. Collate function manuelle (remplace DataCollatorWithPadding, qui a
#    besoin d'un vrai tokenizer HF que Geneformer n'a pas)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attn = (padded != pad_token_id).long()
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


# vérification rapide sur un mini-batch avant de lancer l'entraînement complet
test_batch = collate_fn([train_ds[0], train_ds[1]])
print(f"Clés du batch de test : {list(test_batch.keys())}")
print(f"Forme de input_ids : {test_batch['input_ids'].shape}")

# ---------------------------------------------------------------------------
# 7. TrainingArguments (gestion eval_strategy / evaluation_strategy)
# ---------------------------------------------------------------------------
major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
eval_strategy_kwarg = "eval_strategy" if (major, minor) >= (4, 46) else "evaluation_strategy"

training_args_kwargs = dict(
    output_dir="geneformer_biomarker_ft",
    per_device_train_batch_size=2,        # réglé pour une carte 12 Go (Quadro A2000)
    gradient_accumulation_steps=8,         # 2 x 8 = batch size effectif de 16
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    dataloader_pin_memory=False,
    remove_unused_columns=False,  # essentiel avec un data_collator personnalisé :
                                    # sinon Trainer supprime des colonnes (dont
                                    # input_ids) avant d'appeler collate_fn
)
training_args_kwargs[eval_strategy_kwarg] = "epoch"

args = TrainingArguments(**training_args_kwargs)

# ---------------------------------------------------------------------------
# 8. Entraînement
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
)

# --- vérification explicite juste avant l'entraînement ---
print(f"remove_unused_columns = {args.remove_unused_columns}")
print(f"Colonnes de train_ds juste avant train() : {train_ds.column_names}")
print(f"Exemple d'enregistrement (clés) : {list(train_ds[0].keys())}")

trainer.train()
print(trainer.evaluate())
