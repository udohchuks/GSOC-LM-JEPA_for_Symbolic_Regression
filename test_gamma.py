from tokenizer.tokenize import test_roundtrip
print("Testing gamma equation:")
test_roundtrip("1/(gamma-1)*pr*V", verbose=True)
print("\nTesting beta equation:")
test_roundtrip("beta*(1+alpha*cos(theta))", verbose=True)
