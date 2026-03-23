"""
RPN Tokenizer for LLM-JEPA Symbolic Regression.

Manages the closed vocabulary of ~35 tokens and provides
utilities for converting between symbolic expressions and
integer index sequences.

Design decisions:
    - RPN (postfix) notation: eliminates parentheses,
      enables grammar-constrained generation via stack counter
    - Fixed closed vocabulary: ~35 tokens, no unknown tokens
      in physics equations
    - Stack counter validation: O(1) per token, no parser needed
"""
from __future__ import annotations
from typing import List, Optional
import sympy

# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = '<PAD>'   # padding to fixed length
BOS_TOKEN = '<BOS>'   # beginning of sequence
EOS_TOKEN = '<EOS>'   # end of sequence
UNK_TOKEN = '<UNK>'   # unknown

# ── Variable tokens ───────────────────────────────────────────────────────────
# Up to 9 variables covers all Feynman equations (max is 9 variables)
# Generic names x1-x9 are used regardless of the actual variable name
# (theta, G, m1, etc.) — the unit embedder carries the physical identity
VARIABLE_TOKENS = [f'x{i}' for i in range(1, 10)]

# ── Constants ─────────────────────────────────────────────────────────────────
# Simple rationals and mathematical constants that appear in physics formulas
# Arbitrary numerical constants are handled by BFGS post-processing at inference
CONSTANT_TOKENS = ['0', '1', '2', '3', 'pi', 'e',
                   'c1', 'c2', 'c3', 'c4', 'c5']
# ── Binary operators ──────────────────────────────────────────────────────────
BINARY_TOKENS = ['+', '-', '*', '/']

# ── Unary operators ───────────────────────────────────────────────────────────
UNARY_TOKENS = [
    'sqrt',     'sq',
    'exp',      'log',
    'sin',      'cos',
    'tan',      'arcsin', 
    'arccos',   'arctan',
    'inv',      'abs',
    'neg',
]

# ── Full vocabulary in fixed order ────────────────────────────────────────────
VOCAB = (
    [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] +
    VARIABLE_TOKENS + CONSTANT_TOKENS + BINARY_TOKENS + UNARY_TOKENS
)


# Bidirectional lookup tables
TOKEN2IDX: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}
IDX2TOKEN: dict[int, str] = {idx:tok for tok, idx in TOKEN2IDX.items()}

VOCAB_SIZE = len(VOCAB)
PAD_IDX = TOKEN2IDX[PAD_TOKEN]
EOS_IDX = TOKEN2IDX[EOS_TOKEN]
BOS_IDX = TOKEN2IDX[BOS_TOKEN]
UNK_IDX = TOKEN2IDX[UNK_TOKEN]
MAX_SEQ_LEN = 40


# ── Operator arity ────────────────────────────────────────────────────────────
# Arity = number of arguments an operator consumes
# This drives the stack counter validation logic
ARITY: dict[str, int] = {}
for tok in BINARY_TOKENS:
    ARITY[tok] = 2
for tok in UNARY_TOKENS:
    ARITY[tok] = 1
for tok in VARIABLE_TOKENS + CONSTANT_TOKENS:
    ARITY[tok] = 0   # leaves push onto stack, consume nothing


def is_valid_rpn(tokens: List[str]) -> bool:
    """
    Check if a token sequence forms a valid RPN expression.
    
    Uses stack counter — O(n) total, O(1) per token.
    """
    depth = 0
    for tok in tokens:
        # Skip special tokens
        if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
            continue
        arity = ARITY.get(tok, 0)

        if arity == 2:
            if depth < 2:
                return False  # not enough operands
            depth -= 1 # consume 2, push 1 → net -1
        elif arity == 1:
            if depth < 1:
                return False
        else:
            depth += 1 # leaf: push 
    return depth == 1          # exactly one complete expression


def get_valid_next_tokens(stack_depth: int,
                          seq_len: int,
                          max_len: int) -> List[int]:
    """
    Return vocab indices of tokens valid at the current generation step.
    
    This is the validity mask applied before softmax during inference.
    Invalid tokens are set to -inf, making their probability exactly 0.
    
    Args:
        stack_depth: current expression stack depth
        seq_len:     tokens generated so far (excluding BOS)
        max_len:     maximum sequence length
    
    Returns:
        List of valid token indices.
    """
    valid = []
    
    # If we are at the last step, we must end if possible
    if seq_len >= max_len - 1:
        if stack_depth == 1:
            valid.append(EOS_IDX)
        return valid

    for idx, tok in IDX2TOKEN.items():
        # Skip special tokens that shouldn't be generated in the middle of a formula
        if tok in (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN):
            continue

        if tok == EOS_TOKEN:
            if stack_depth == 1:
                valid.append(idx)
            continue
            
        arity = ARITY.get(tok, 0)

        if arity == 2:
            if stack_depth >= 2:
                valid.append(idx)
        elif arity == 1:
            if stack_depth >= 1:
                valid.append(idx)
        else:
            valid.append(idx)
            
    return valid

def encode_formula(
    tokens: List[str],
    add_bos: bool = True,
    add_eos: bool = True,
    pad_to: Optional[int] = None ) -> List[int]:
    """
    Convert RPN token list to padded integer index sequence.
    
    Args:
        tokens:   list of RPN token strings (without BOS/EOS)
        add_bos:  prepend BOS index
        add_eos:  append EOS index  
        pad_to:   if set, pad or truncate to this length
    
    Returns:
        List of integer indices.
    """
    seq = []
    if add_bos:
        seq.append(BOS_IDX)
    seq.extend(TOKEN2IDX.get(t, UNK_IDX) for t in tokens)
    if add_eos:
        seq.append(EOS_IDX)
    if pad_to:
        if len(seq) > pad_to:
            return seq[:pad_to]
        else:
            seq = seq +  [PAD_IDX] *(pad_to - len(seq))
    return seq

def decode_formula(indices: List[int],
                   strip_special: bool = True) -> List[str]:
    """
    Convert integer index sequence back to RPN token list.
    """
    tokens = [IDX2TOKEN.get(id, UNK_IDX) for id in indices]
    if strip_special:
        tokens = [t for t in tokens if t not in (PAD_TOKEN, EOS_TOKEN, BOS_TOKEN)]
    return tokens


def formula_string_to_rpn(formula_str: str,
                            var_names: List[str]) -> List[str]:
    """
    Parse a formula string from the AIF CSV and convert to RPN tokens.
    
    Args:
        formula_str: e.g. "exp(-theta**2/2)/sqrt(2*pi)"
        var_names:   variable names in order, e.g. ['theta']
                     These map to x1, x2, ... in the RPN output
    
    Returns:
        List of RPN token strings.
    """
    # Build variable name → RPN token mapping
    # theta → x1, G → x2, m1 → x3, etc.
    var_map = {name: f'x{i+1}' for i, name in enumerate(var_names)}
    # Tell SymPy which names are variables vs functions
    local_dict = {name: sympy.Symbol(name) for name in var_names}
    local_dict['pi'] = sympy.pi
    local_dict['e']  = sympy.E
    try:
        expr = sympy.sympify(formula_str, locals=local_dict)
        expr = sympy.simplify(expr)
    except Exception as exc:
        raise ValueError(
            f"Cannot parse formula: {formula_str}"
        ) from exc
    return _sympy_to_rpn(expr, var_map)


def _sympy_to_rpn(expr: sympy.Expr,
                   var_map: dict[str, str]) -> List[str]:
    """
    Recursively convert a SymPy expression to RPN tokens.
    Postorder traversal: children before parent.
    """
    tokens: List[str] = []
    const_counter = [0]
    _traverse(expr, var_map, tokens, const_counter)
    return tokens

def _traverse(expr, var_map, tokens, const_counter):
    """Recursive postorder traversal."""
    # ── Variables ─────────────────────────────────────────────────────────────
    if isinstance(expr, sympy.Symbol):
        name = str(expr)
        if name in var_map:
            tokens.append(var_map[name])
        else:
            tokens.append(UNK_TOKEN)
        return
    # ── Integer constants ─────────────────────────────────────────────────────
    if isinstance(expr, sympy.Integer):
        val = str(int(expr))
        tokens.append(val if val in TOKEN2IDX else _next_const(const_counter))
        return
    # ── pi and e ──────────────────────────────────────────────────────────────
    if expr == sympy.pi:
        tokens.append('pi')
        return
    if expr == sympy.E:
        tokens.append('e')
        return
    # ── Float: round to nearest integer constant ──────────────────────────────
    if isinstance(expr, sympy.Float):
        val = float(expr)
        rounded = str(int(round(val)))
        tokens.append(_next_const(const_counter))
        return
    
    # ── Rational: express as numerator / denominator ──────────────────────────
    if isinstance(expr, sympy.Rational):
        _traverse(sympy.Integer(expr.p), var_map, tokens, const_counter)
        _traverse(sympy.Integer(expr.q), var_map, tokens, const_counter)
        tokens.append('/')
        return
    # ── Negation: -x → x neg ──────────────────────────────────────────────────
    if isinstance(expr, sympy.Mul) and expr.args[0] == sympy.Integer(-1):
        inner = sympy.Mul(*expr.args[1:])
        _traverse(inner, var_map, tokens, const_counter)
        tokens.append('neg')
        return
    # ── Addition: a+b+c → a b + c + ───────────────────────────────────────────
    if isinstance(expr, sympy.Add):
        args = list(expr.args)
        _traverse(args[0], var_map, tokens, const_counter)
        for arg in args[1:]:
            _traverse(arg, var_map, tokens, const_counter)
            tokens.append('+')
        return
    
    # ── Multiplication: a*b*c → a b * c * ────────────────────────────────────
    if isinstance(expr, sympy.Mul):
        args = list(expr.args)
        _traverse(args[0], var_map, tokens, const_counter)
        for arg in args[1:]:
            _traverse(arg, var_map, tokens, const_counter)
            tokens.append('*')
        return
    # ── Powers ────────────────────────────────────────────────────────────────
    if isinstance(expr, sympy.Pow):
        base, exp_val = expr.args
        if exp_val == sympy.Integer(2):
            _traverse(base, var_map, tokens, const_counter)
            tokens.append('sq')
        elif exp_val == sympy.Integer(-1):
            _traverse(base, var_map, tokens, const_counter)
            tokens.append('inv')
        elif exp_val == sympy.Rational(1, 2):
            _traverse(base, var_map, tokens, const_counter)
            tokens.append('sqrt')
        elif exp_val == sympy.Rational(-1, 2):
            _traverse(base, var_map, tokens, const_counter)
            tokens.append('sqrt')
            tokens.append('inv')
        else:
            # Fallback for other powers
            _traverse(base, var_map, tokens, const_counter)
            _traverse(exp_val, var_map, tokens, const_counter)
            tokens.append('*')
        return
    # ── Standard unary functions ──────────────────────────────────────────────
    SYMPY_UNARY_MAP = {
        'sqrt':  'sqrt',
        'exp':   'exp',
        'log':   'log',
        'sin':   'sin',
        'cos':   'cos',
        'tan':   'tan',
        'asin':  'arcsin',
        'acos':  'arccos',
        'atan':  'arctan',
        'Abs':   'abs',
    }
    func_name = type(expr).__name__
    if func_name in SYMPY_UNARY_MAP:
        _traverse(expr.args[0], var_map, tokens, const_counter)
        tokens.append(SYMPY_UNARY_MAP[func_name])
        return
    
    # ── Fallback ──────────────────────────────────────────────────────────────
    raise ValueError(f"Unsupported expression: {type(expr).__name__}: {expr}")

def _next_const(counter: List[int]) -> str:
    """
    Return the next available constant placeholder token.
    counter is a mutable list so recursive calls share state.
    """
    MAX_CONSTS = 5
    idx = counter[0] % MAX_CONSTS + 1   # cycles c1...c5
    counter[0] += 1
    return f'c{idx}'


def rpn_to_sympy(tokens: List[str]) -> sympy.Expr:
    """
    Convert a list of RPN tokens back into a SymPy expression.
    This preserves specific tokens like 'c1', 'x1', etc. 
    as SymPy Symbols.
    """
    # Inverse mappings for unary/binary operations
    SYMPY_UNARY_INV = {
        'sqrt':   sympy.sqrt,
        'sq':     lambda x: x**2,
        'exp':    sympy.exp,
        'log':    sympy.log,
        'sin':    sympy.sin,
        'cos':    sympy.cos,
        'tan':    sympy.tan,
        'arcsin': sympy.asin,
        'arccos': sympy.acos,
        'arctan': sympy.atan,
        'inv':    lambda x: 1/x,
        'abs':    sympy.Abs,
        'neg':    lambda x: -x,
    }

    SYMPY_BINARY_INV = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b,
    }

    stack = []
    
    for tok in tokens:
        if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            continue
            
        arity = ARITY.get(tok, 0)
        
        if arity == 0:
            if tok == 'pi':
                stack.append(sympy.pi)
            elif tok == 'e':
                stack.append(sympy.E)
            elif tok in ['0', '1', '2', '3']:
                stack.append(sympy.Integer(tok))
            else:
                # Variables (x1..x9) and constants (c1..c5) map to Symbols
                stack.append(sympy.Symbol(tok))
                
        elif arity == 1:
            if len(stack) < 1:
                raise ValueError(f"Not enough arguments for unary operator {tok}")
            arg = stack.pop()
            func = SYMPY_UNARY_INV.get(tok)
            if not func:
                raise ValueError(f"Unknown unary operator {tok}")
            stack.append(func(arg))
            
        elif arity == 2:
            if len(stack) < 2:
                raise ValueError(f"Not enough arguments for binary operator {tok}")
            right = stack.pop()
            left = stack.pop()
            func = SYMPY_BINARY_INV.get(tok)
            if not func:
                raise ValueError(f"Unknown binary operator {tok}")
            stack.append(func(left, right))
            
    if len(stack) != 1:
        raise ValueError("Invalid RPN sequence: stack does not contain exactly 1 element at the end")
        
    return stack[0]

# Quick test at bottom of file (or in a test script)
if __name__ == '__main__':
    # Test 1: valid and invalid RPN
    assert is_valid_rpn(['x1', 'x2', '+'])
    assert not is_valid_rpn(['+', 'x1', 'x2'])
    assert not is_valid_rpn(['x1', 'x2', '+', 'x3'])
    print('Stack counter: OK')
    
    # Test 2: validity mask
    valid = get_valid_next_tokens(stack_depth=0, seq_len=0, max_len=25)
    for idx in valid:
        assert ARITY.get(IDX2TOKEN[idx], 0) == 0, \
            f'Binary/unary at depth 0: {IDX2TOKEN[idx]}'
    print('Validity mask at depth 0: OK')
    
    # Test 3: formula tokenisation
    rpn = formula_string_to_rpn(
        'exp(-theta**2/2)/sqrt(2*pi)', ['theta']
    )
    print(f'Gaussian RPN: {rpn}')
    assert is_valid_rpn(rpn)
    print('Formula tokenisation: OK')
    
    # Test 4: encode/decode round trip
    encoded = encode_formula(rpn, pad_to=MAX_SEQ_LEN)
    assert len(encoded) == MAX_SEQ_LEN
    decoded = decode_formula(encoded)
    assert decoded == rpn
    print('Encode/decode round trip: OK')
    
    print(f'\nVocab size: {VOCAB_SIZE}')
    print('All tokenizer tests passed.')
    