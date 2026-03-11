import sympy

OP_MAP = {
    'add': '+',
    'pow': '**',
    'mul': '*'
}

known_functions = ['sin', 'cos', 'tan', 'acos', 'asin', 'atan', 'log', 'exp',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'abs'
                    ]

CONTROL_TOKENS = ['[SOS]', '[EOS]', '[PRED]', '[MASK]', '[PAD]']

SPECIAL_TOKENS = list(CONTROL_TOKENS) +  ['[NUM]']

