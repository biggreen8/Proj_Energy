import random
import os

global n_digits
n_digits = 8

amount = 500

def generate_n_by_x_problems(n, x, count, direction=True):
    problems = []
    for _ in range(count):
        a = random.randint(10**(n-1), 10**n - 1)
        b = random.randint(10**(x-1), 10**x - 1)
        if direction:
            problems.append(f"{a} + {b} = ")
        else:
            problems.append(f"{b} + {a} = ")
    return problems


if __name__ == "__main__":
    for k in range(1, n_digits+1):
        for h in range(1, k+1):
            # Define the directory name
            n_dig_folder = f"{k}_problems"
            
            # Ensure the directory exists
            os.makedirs(n_dig_folder, exist_ok=True)
            
            # Generate and write the first set of problems
            problems = generate_n_by_x_problems(k, h, amount)
            with open(f"{n_dig_folder}/{k}_by_{h}_problems.txt", "w") as f:
                f.write("\n".join(problems))
            
            # Generate and write the second set of problems
            problems = generate_n_by_x_problems(k, h, amount, False)
            with open(f"{n_dig_folder}/{h}_by_{k}_problems.txt", "w") as g:
                g.write("\n".join(problems))
