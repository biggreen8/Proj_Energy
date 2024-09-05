import random

global n_digits
n_digits = 7

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
            n_dig_folder= f"{k}_problems"
            problems = generate_n_by_x_problems(n_digits, h, amount)
            f = open(f"{k}_by_{h}_problems.txt", "w")
            f.write("\n".join(problems))
            problems = generate_n_by_x_problems(n_digits, h, amount, False)
            f = open(f"{h}_by_{k}_problems.txt", "w")
            f.write("\n".join(problems))

   