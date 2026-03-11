"""
tokenizer/vocab.py

Builds vocabulary from the Feynman dataset.
Maps every token string to a unique integer ID and back.

Vocabulary structure:
  [0]  [PAD]   - padding token
  [1]  [SOS]   - start of sequence
  [2]  [EOS]   - end of sequence
  [3]  [PRED]  - JEPA predictor token
  [4]  [MASK]  - masked token
  [5]  [NUM]   - float constant placeholder
  [6+] math tokens: operators, functions, integers, fractions, variables
"""

import json
import sys
import os
import pandas as pd 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.run_tokenizer import formula_to_tokens

SPECIAL_TOKENS = ['[PAD]', '[SOS]', '[EOS]', '[PRED]', '[MASK]', '[NUM]']

PAD_ID  = 0
SOS_ID  = 1
EOS_ID  = 2
PRED_ID = 3
MASK_ID = 4
NUM_ID  = 5

def build_vocab(csv_path: str) -> tuple[dict, dict]:
    """
    Scan every equation in the dataset and collect all tokens.
    Build tok2id and id2tok mappings.

    Returns:
        tok2id: token string → integer ID
        id2tok: integer ID  → token string
    """
    df = pd.read_csv(csv_path)

    all_tokens = set()

    for _, row in df.iterrows():
        formula = str(row['Formula'])

        tokens, _, _, _ = formula_to_tokens(formula)
        all_tokens.update(tokens)

    math_tokens = all_tokens - set(SPECIAL_TOKENS)
    math_tokens_sorted = sorted(math_tokens)

    all_tokens =  SPECIAL_TOKENS + math_tokens_sorted 

    tok2id = {tok: idx for idx,tok in enumerate(all_tokens)}
    id2tok = {idx: tok for idx, tok in enumerate(all_tokens)}

    return tok2id, id2tok


def save_vocab(tok2id: dict, id2tok: dict, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'tok2id.json'), 'w') as f:
        json.dump(tok2id, f, indent=2)
    
    id2tok_str = {str(k): v for k, v in id2tok.items()}

    with open(os.path.join(save_dir, 'id2tok.json'), 'w') as f:
        json.dump(id2tok_str, f, indent=2)
    
    print(f"Vocabulary saved to {save_dir}/")

def load_vocab(save_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(save_dir, 'tok2id.json'),'r') as f:
        tok2id = json.load(f)
    with open(os.path.join(save_dir, 'id2tok.json'),'r') as f:
        id2tok_str = json.load(f)
        id2tok = {int(k): v for k, v in id2tok_str.items()}

    return tok2id, id2tok

def print_vocab_summary(tok2id: dict):
    print("=" * 45)
    print("VOCABULARY SUMMARY")
    print("=" * 45)
    print(f"Total size      : {len(tok2id)}")
    print(f"Special tokens  : {SPECIAL_TOKENS}")
    print()

    # Group math tokens by type
    operators  = [t for t in tok2id if t in ('+', '-', '*', '/', '**')]
    functions  = [t for t in tok2id if t in
                  ('sin','cos','tan','exp','log','sqrt','abs',
                   'sinh','cosh','tanh','asin','acos','atan','arcsin')]
    integers   = [t for t in tok2id if t not in SPECIAL_TOKENS
                  and t not in operators and t not in functions
                  and t.lstrip('-').isdigit()]
    fractions  = [t for t in tok2id if '/' in t and t not in operators]
    variables  = [t for t in tok2id if t.startswith('v') and
                  t[1:].isdigit()]
    constants  = [t for t in tok2id if t in ('pi',)]

    print(f"Operators       : {sorted(operators)}")
    print(f"Functions       : {sorted(functions)}")
    print(f"Integers        : {sorted(integers)}")
    print(f"Fractions       : {sorted(fractions)}")
    print(f"Variables       : {sorted(variables)}")
    print(f"Constants       : {sorted(constants)}")
    print()
    print("Full tok2id mapping:")
    for tok, idx in sorted(tok2id.items(), key=lambda x: x[1]):
        print(f"  {idx:3d}  {tok}")

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'FeynmanEquations.csv')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')

    tok2id, id2tok = build_vocab(csv_path)
    print_vocab_summary(tok2id)
    save_vocab(tok2id, id2tok, save_dir)

    # Verify load works
    tok2id_loaded, id2tok_loaded = load_vocab(save_dir)
    assert tok2id_loaded == tok2id
    print("\nLoad verification: PASSED")