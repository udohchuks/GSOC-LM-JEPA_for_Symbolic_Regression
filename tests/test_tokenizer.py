"""
Tests for data/tokenizer.py

Tests RPN tokenization, validity checking, and encode/decode operations.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.tokenizer import (
    TOKEN2IDX, IDX2TOKEN, VOCAB_SIZE, MAX_SEQ_LEN,
    encode_formula, decode_formula, is_valid_rpn,
    get_valid_next_tokens, formula_string_to_rpn, rpn_to_sympy,
    ARITY, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
)


class TestRPNValidity:
    """Test RPN validity checking."""
    
    def test_valid_simple_expression(self):
        """Test valid simple expressions."""
        assert is_valid_rpn(['x1', 'x2', '+']) == True
        assert is_valid_rpn(['x1', 'x2', '*']) == True
        assert is_valid_rpn(['x1', 'sin']) == True
        
    def test_invalid_expressions(self):
        """Test invalid expressions are caught."""
        # Operator before operands
        assert is_valid_rpn(['+', 'x1', 'x2']) == False
        # Incomplete expression (stack != 1 at end)
        assert is_valid_rpn(['x1', 'x2', '+', 'x3']) == False
        # Not enough operands for binary op
        assert is_valid_rpn(['x1', '+']) == False
        
    def test_valid_complex_expression(self):
        """Test valid complex expressions."""
        # sin(x1 + x2) -> x1 x2 + sin
        assert is_valid_rpn(['x1', 'x2', '+', 'sin']) == True
        # (x1 + x2) * (x3 + x4) -> x1 x2 + x3 x4 + *
        assert is_valid_rpn(['x1', 'x2', '+', 'x3', 'x4', '+', '*']) == True


class TestValidityMask:
    """Test get_valid_next_tokens function."""
    
    def test_depth_zero_only_leaves(self):
        """At depth 0, only leaves (variables, constants) are valid."""
        valid = get_valid_next_tokens(stack_depth=0, seq_len=0, max_len=25)
        for idx in valid:
            tok = IDX2TOKEN[idx]
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                continue
            arity = ARITY.get(tok, 0)
            assert arity == 0, f"At depth 0, only leaves allowed, got {tok} (arity={arity})"
            
    def test_depth_one_allows_unary_and_leaves(self):
        """At depth 1, unary operators and leaves are valid."""
        valid = get_valid_next_tokens(stack_depth=1, seq_len=0, max_len=25)
        valid_toks = [IDX2TOKEN[idx] for idx in valid if IDX2TOKEN[idx] not in (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN)]
        
        # Should include leaves
        assert 'x1' in valid_toks
        # Should include unary operators
        assert 'sin' in valid_toks
        # Should include EOS (depth=1 means we can end)
        assert EOS_TOKEN in valid_toks
        
    def test_depth_two_requires_binary_or_unary(self):
        """At depth 2, binary operators become valid."""
        valid = get_valid_next_tokens(stack_depth=2, seq_len=0, max_len=25)
        valid_toks = [IDX2TOKEN[idx] for idx in valid if IDX2TOKEN[idx] not in (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN)]
        
        # Should include binary operators
        assert '+' in valid_toks
        assert '*' in valid_toks


class TestEncodeDecode:
    """Test encode/decode round-trip."""
    
    def test_round_trip_simple(self):
        """Test simple expression round-trip."""
        tokens = ['x1', 'x2', '+']
        encoded = encode_formula(tokens, pad_to=MAX_SEQ_LEN)
        decoded = decode_formula(encoded)
        assert decoded == tokens
        
    def test_round_trip_complex(self):
        """Test complex expression round-trip."""
        tokens = ['x1', 'x2', '+', 'sin', 'c1', '*', 'c2', '+']
        encoded = encode_formula(tokens, pad_to=MAX_SEQ_LEN)
        decoded = decode_formula(encoded)
        assert decoded == tokens
        
    def test_encode_adds_bos_eos(self):
        """Test that encode adds BOS and EOS tokens."""
        tokens = ['x1', 'x2', '+']
        encoded = encode_formula(tokens, add_bos=True, add_eos=True, pad_to=None)
        # First should be BOS, last should be EOS
        assert encoded[0] == TOKEN2IDX[BOS_TOKEN]
        assert encoded[-1] == TOKEN2IDX[EOS_TOKEN]


class TestFormulaToRPN:
    """Test formula_string_to_rpn function."""
    
    def test_gaussian_rpn(self):
        """Test Gaussian formula tokenization."""
        rpn = formula_string_to_rpn('exp(-theta**2/2)/sqrt(2*pi)', ['theta'])
        assert is_valid_rpn(rpn)
        
    def test_simple_linear(self):
        """Test simple linear formula."""
        rpn = formula_string_to_rpn('x1 + x2', ['x1', 'x2'])
        assert is_valid_rpn(rpn)
        # Convert back to sympy and verify
        expr = rpn_to_sympy(rpn)
        assert str(expr) == 'x1 + x2' or str(expr) == 'x2 + x1'  # Order may vary
        
    def test_multiplication(self):
        """Test multiplication formula."""
        rpn = formula_string_to_rpn('x1 * x2', ['x1', 'x2'])
        assert is_valid_rpn(rpn)


class TestRPNSympyConversion:
    """Test RPN to SymPy conversion."""
    
    def test_addition(self):
        """Test addition conversion."""
        rpn = ['x1', 'x2', '+']
        expr = rpn_to_sympy(rpn)
        assert str(expr) in ['x1 + x2', 'x2 + x1']
        
    def test_nested_operations(self):
        """Test nested operations."""
        # (x1 + x2) * x3 -> x1 x2 + x3 *
        rpn = ['x1', 'x2', '+', 'x3', '*']
        expr = rpn_to_sympy(rpn)
        # Should be equivalent to (x1 + x2) * x3
        assert 'x1' in str(expr) and 'x2' in str(expr) and 'x3' in str(expr)
        
    def test_constants(self):
        """Test constant placeholders."""
        rpn = ['c1', 'x1', '*', 'c2', '+']
        expr = rpn_to_sympy(rpn)
        assert 'c1' in str(expr) and 'c2' in str(expr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
