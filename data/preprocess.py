import pandas as pd
import sys
import os

# Add project root to path so we can import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.run_tokenizer import formula_to_tokens, test_roundtrip

def load_equations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} equations")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 3 rows:")
    print(df.head(3))
    return df


def run_tokenizer_on_all(df: pd.DataFrame):
    """
    Run tokenizer on every equation.
    Report:
      - How many pass round-trip
      - How many fail and why
      - Full vocabulary across all equations
      - Sequence length distribution
    """
    passed   = []
    failed   = []
    all_tokens = set()
    all_lengths = []

    print("\n" + "="*55)
    print("Running tokenizer on all equations...")
    print("="*55)

    for _, row in df.iterrows():
        formula = str(row['Formula'])

         # Skip empty rows
        if formula == 'nan' or formula.strip() == '':
            continue
        try:
            tokens, _, _, _ = formula_to_token(formula)
            if 'imaginaryunit' in tokens:
                print(f"CULPRIT: {formula}")
                print(f"Tokens: {tokens}")
        except:
           pass
        # Try tokenizing
        try:
            tokens, var_map, var_map_inv, num_values = formula_to_tokens(formula)
            all_tokens.update(tokens)
            all_lengths.append(len(tokens))
        except Exception as e:
            failed.append((formula, f"TOKENIZE ERROR: {e}"))
            continue

        # Try round-trip
        ok = test_roundtrip(formula, verbose=False)
        if ok:
            passed.append(formula)
        else:
            failed.append((formula, "ROUND-TRIP FAIL"))

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\nTotal equations : {len(df)}")
    print(f"Passed          : {len(passed)}")
    print(f"Failed          : {len(failed)}")
    print()

    if failed:
        print("FAILURES:")
        for formula, reason in failed:
            print(f"  {reason}: {formula}")
        print()

    print("VOCABULARY:")
    print(f"  Unique tokens : {len(all_tokens)}")
    print(f"  All tokens    : {sorted(all_tokens)}")
    print()

    import numpy as np
    lengths = all_lengths
    print("SEQUENCE LENGTHS:")
    print(f"  Min    : {min(lengths)}")
    print(f"  Max    : {max(lengths)}")
    print(f"  Mean   : {np.mean(lengths):.1f}")
    print(f"  Median : {np.median(lengths):.1f}")
    print(f"  95th % : {np.percentile(lengths, 95):.0f}")
    print()

    return passed, failed, all_tokens, all_lengths


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw', 'FeynmanEquations.csv')
    df       = load_equations(csv_path)
    passed, failed, vocab, lengths = run_tokenizer_on_all(df)