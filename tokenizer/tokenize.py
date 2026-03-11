import sympy
import sys
import os
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.vocab import known_functions, CONTROL_TOKENS,SPECIAL_TOKENS, OP_MAP

from sympy import (
    Add, Mul, Pow, Symbol, Integer, Float, Rational,
    sin, cos, tan, exp, exp, log, sqrt, Abs,
    sinh, cosh, tanh, asin, acos, atan
)


def binarise(expr):
    """
    Force all Mul and Add nodes to be strictly binary (two children).

    Sympy represents a*b*c as Mul(a, b, c) — three children.
    Our decoder assumes binary operators so we fold left:
        Mul(a, b, c)  →  Mul(Mul(a, b), c)
        Add(a, b, c)  →  Add(Add(a, b), c)

    Must be called BEFORE preorder_walk.
    """

    if expr.is_Atom:
        return expr
    
    args = [binarise(a) for a in expr.args]
    # Mul
    if isinstance(expr, Mul) and len(args) > 2:
        result = Mul(args[0], args[1], evaluate=False)

        for a in args[2:]:
            result = Mul(result, a, evaluate=False)
        return result
    # Add 
    if isinstance(expr, Add) and len(args) > 2:
        result = Add(args[0], args[1], evaluate=False)
        for a in args[2:]:
            result = Add(result, a, evaluate=False)

        return result
    return expr.func(*args, evaluate=False)

def preorder_walk(expr) -> list:
    """
    Walk a sympy expression tree in preorder.
    Emits the operator/function token FIRST, then recurses into children.

    Returns a flat list of string tokens.

    Preorder means:
        For a node N with children A and B:
        emit N, then recursively emit A, then recursively emit B

    Example: Mul(Add(v0, v1), v2)
        → ['*', '+', 'v0', 'v1', 'v2']
          (root * first, then left subtree +,v0,v1, then right leaf v2)
    """
    if isinstance(expr, Symbol):
        return [str(expr)]
    if isinstance(expr, Integer):
        return [str(int(expr))]
    if isinstance(expr, Rational):
        return [f"{expr.p}/{expr.q}"]
    if isinstance(expr, Float):
        return ['[NUM]']
    op = type(expr).__name__.lower()
    args = expr.args
    token = OP_MAP.get(op, op) # e.g add -> + , sin -> sin
    result = [token]

    for arg in args:
        result.extend(preorder_walk(arg))
    return result


def _collect_num_values(expr) -> list:
  """
    Collect all floating-point/rational values from the expression tree
    in the same order that preorder_walk emits [NUM] tokens.
  """
  if isinstance(expr, (Float)):
    return [float(expr)]
  if expr.is_Atom:
    return []
  result = []
  for child in expr.args:
    result.extend(_collect_num_values(child))
  return result

def formula_to_tokens(formula_str: str) -> tuple:
    """
    Main entry point. Convert a formula string to preorder tokens.

    Args:
        formula_str: raw formula string e.g. "G*m1*m2/r**2"

    Returns:
        tokens:      list of strings e.g. ['*','*','*','v0','v1','v2','**','v3','-2']
        var_map:     original → placeholder  e.g. {'G':'v0','m1':'v1','m2':'v2','r':'v3'}
        var_map_inv: placeholder → original  e.g. {'v0':'G','v1':'m1','v2':'m2','v3':'r'}
        num_values:  list of float values for each [NUM] token in order
                     e.g. [0.5] if the formula contained 0.5

    Raises:
        ValueError if sympy cannot parse the formula
    """
    try:
        RESERVED_NAMES = {
        'gamma': sympy.Symbol('gamma'),
        'beta':  sympy.Symbol('beta'),
        'alpha': sympy.Symbol('alpha'),
        'zeta':  sympy.Symbol('zeta'),
        'delta': sympy.Symbol('delta'),
        'Lambda': sympy.Symbol('Lambda'),
        }
        expr = parse_expr(formula_str, local_dict=RESERVED_NAMES,
                  transformations=standard_transformations)
    except Exception as e:
        raise ValueError(f"sympy could not parse '{formula_str}': {e}")
    variables = sorted(expr.free_symbols, key=lambda s: s.name)
    # remove pi if present
    variables = [v for v in variables if str(v) != 'pi']

    var_map = {str(v): f"v{i}" for i, v in enumerate(variables)}
    var_map_inv = {f"v{i}": str(v) for i, v in enumerate(variables)}

    substitution = {v: Symbol(f"v{i}") for i, v in enumerate(variables)}

    canonical_expr = expr.subs(substitution)

    binary_expr = binarise(canonical_expr)

    tokens = preorder_walk(binary_expr)

    num_values = _collect_num_values(binary_expr)

    return tokens, var_map, var_map_inv, num_values

def _parse_preorder(token_iter, num_counter=None) -> sympy.Basic:
    """
    Recursive preorder parser — rebuilds sympy tree from token sequence.

    Reads one token, determines what kind of node it is,
    then recursively reads exactly as many children as that node needs.

    Binary operators need 2 children.
    Unary functions need 1 child.
    Leaves (numbers, variables) need 0 children.


    num_counter - counts the number of [NUM] variable
    """

    if num_counter is None:
        num_counter = [0]
    
    try:
        token = next(token_iter)
    except StopIteration:
        raise ValueError("Unexpected end of token sequence")
    if token == '[NUM]':
        scalar = f"C_{num_counter[0]}"
        num_counter[0] += 1
        return sympy.Symbol(scalar)
    try:
        return sympy.Integer(int(token))
    except:
        pass
    try:
        return sympy.Float(float(token))
    except:
        pass
    if token == 'pi':
        return sympy.pi

    if '/' in token and token != '/':
        try:
            p, q = token.split('/')
            return sympy.Rational(int(p), int(q))
        except:
            pass
    if token in known_functions:
        val = _parse_preorder(token_iter, num_counter)
        return getattr(sympy, token)(val)

    if token in ('+', '-', '*', '/', '**'):
        left = _parse_preorder(token_iter, num_counter)
        right = _parse_preorder(token_iter, num_counter)

        if token == "+":
            return sympy.Add(left, right, evaluate=False)
        if token == "*":
            return sympy.Mul(left, right, evaluate=False)
        if token == "**":
            return sympy.Pow(left, right, evaluate=False)
        if token == '/':
            return sympy.Pow(left, sympify.Pow(right, -1, evaluate=False), evaluate=False)
        if token == '-':
            return sympy.Add(left, sympy.Mul(right, -1, evaluate=False), evaluate=False)
        
        # If we reach here the token is a variable name (G, m1, theta etc.)
    return sympy.Symbol(token)

def tokens_to_formula(tokens: list, var_map_inv: dict) -> str:
  # Strip control tokens — [SOS], [EOS], [PAD] etc. are not math
  clean = [t for t in tokens if t not in CONTROL_TOKENS]
  restored = [var_map_inv.get(t, t) for t in clean]
  try:
    expr = _parse_preorder(iter(restored))
    return str(expr)
  except: 
    return '' # broken tree e.g : [+, 2, [PAD]] -> [+, 2] -> [+, 2, ' ']

def test_roundtrip(formula_str: str, verbose: bool = False) -> bool:
  """
  Test tokenize → detokenize round-trip by numerical evaluation.
  
  Plugs in concrete numbers for all variables and checks that
  the original and recovered expressions give the same output.
  Uses multiple test points to avoid accidental equality.
  """
  try:
    RESERVED_NAMES = {
        'gamma': sympy.Symbol('gamma'),
        'beta':  sympy.Symbol('beta'),
        'alpha': sympy.Symbol('alpha'),
        'zeta':  sympy.Symbol('zeta'),
        'delta': sympy.Symbol('delta'),
        'Lambda': sympy.Symbol('Lambda'),
    }
    print(parse_expr(formula_str, local_dict=RESERVED_NAMES, transformations=standard_transformations))
    # Step 1: Tokenize
    tokens, var_map, var_map_inv, num_values = formula_to_tokens(formula_str)

    # Step 2: Detokenize
    recovered_str = tokens_to_formula(tokens, var_map_inv)

    if not recovered_str:
      if verbose:
        print(f"FAIL (empty recovery): {formula_str}")
      return False
    
    original_expr  = parse_expr(formula_str, local_dict=RESERVED_NAMES, transformations=standard_transformations)
    recovered_expr = parse_expr(recovered_str, local_dict=RESERVED_NAMES, transformations=standard_transformations)

    c_syms = sorted(
        recovered_expr.free_symbols - original_expr.free_symbols,
        key=lambda s: s.name
    )
    #  Constants e.g 3.333, 4.44
    for i, c_sym in enumerate(c_syms):
      val = num_values[i] if i < len(num_values) else 1.0
      recovered_expr = recovered_expr.subs(c_sym, val)
    
    # Union of Variable e.g m, v
    all_vars = sorted(
        original_expr.free_symbols | recovered_expr.free_symbols,
        key=lambda s: s.name
    )
    if not all_vars:
      # No variables — both are constants, compare directly
      passed = abs(float(original_expr) - float(recovered_expr)) < 1e-6
      if verbose:
        print(f"{'PASS' if passed else 'FAIL'}: {formula_str}")
      return passed
    
    test_points = [
        {v: (i+1) * 0.7 for i, v in enumerate(all_vars)},   # point 1
        {v: (i+1) * 1.3 for i, v in enumerate(all_vars)},   # point 2
        {v: (i+1) * 2.1 for i, v in enumerate(all_vars)},   # point 3
    ]

    for point in test_points:
      try:
        orig_val = float(original_expr.subs(point))
        recv_val = float(recovered_expr.subs(point))

        # Check relative difference
        scale = max(abs(orig_val), 1e-10)
        if abs(orig_val - recv_val) / scale > 1e-6:
          if verbose:
            print(f"FAIL: {formula_str}")
            print(f"  tokens:    {tokens}")
            print(f"  var_map:   {var_map}")
            print(f"  recovered: {recovered_str}")
            print(f"  at {point}: original={orig_val}, recovered={recv_val}")
          return False

      except (ValueError, ZeroDivisionError, TypeError):
        # This test point caused a math error (e.g. log of negative)
        # Skip it and try the next point — not a failure
        continue
    if verbose:
      print(f"PASS: {formula_str}")
      print(f"  tokens:    {tokens}")
      print(f"  num_vals:  {num_values}")
      print(f"  var_map:   {var_map}")
      print(f"  recovered: {recovered_str}")

    return True
  except Exception as e:
    if verbose:
      print(f"ERROR: {formula_str} — {e}")
    return False




# # ── Quick test ────────────────────────────────────────────────────────────
# tokens, var_map, inv, num_values = formula_to_tokens('G*m1*m2/r**2')
# print('Tokens:   ', tokens)
# print('Var map:  ', var_map)
# print('Num vals: ', num_values)
# print()

# test_roundtrip('G*m1*m2/r**2', verbose=True)
# print()
# test_roundtrip('exp(-theta**2/2)/sqrt(2*pi)', verbose=True)
# print()
# test_roundtrip('0.5*m*v**2', verbose=True)
# print()
# test_roundtrip('0.5 * m + 0.3 * v', verbose=True)