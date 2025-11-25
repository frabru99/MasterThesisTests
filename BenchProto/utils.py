import difflib

def compareModelArchitecture(model1, model2):
    # Convert models to string representations
    model1_str = str(model1).splitlines()
    model2_str = str(model2).splitlines()

    # Create a differ object
    diff = difflib.ndiff(model1_str, model2_str)

    print(f"Comparing Model 1 vs Model 2:")
    print("-" * 30)
    
    # Print only the differences
    has_diff = False
    for line in diff:
        if line.startswith('+') or line.startswith('-'):
            print(line)
            has_diff = True
    
    if not has_diff:
        print("Architectures are identical.")