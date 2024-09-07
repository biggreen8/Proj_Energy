import random
import os

global n_digits
n_digits = 8

amount = 500

def generate_x_by_y_problems(x, y, count, direction=True):
    """
    Generates "count" number of addition problems of x-digits by y-digits.

    Args:
        x (int): digits in first additive term.
        y (int): digits in second additive term.
        count (int): nubmer of addition problems.
        direction (Boolean): if True: generates x by y digit problems.  If
        False: generates y by x digit problems.

    Returns:
        A list of length "count" where each index is a str of an x-digit by
        y-digit addition problem
    """
    problems = []
    for _ in range(count):
        a = random.randint(10**(x-1), 10**x - 1)
        b = random.randint(10**(y-1), 10**y - 1)
        if direction:
            problems.append(f"{a} + {b} = ")
        else:
            problems.append(f"{b} + {a} = ")
    return problems


if __name__ == "__main__":
    """
    Creates .txt files of "amount" number of all combinations of 1 through n-digit 
    by 1 through n-digit addition problems and stores them in folders titled
    max(x, y)_problems
    """

    for k in range(1, n_digits+1):
        for h in range(1, k+1):
            # Define the directory name
            n_dig_folder = f"{k}_problems"
            
            # Ensure the directory exists
            os.makedirs(n_dig_folder, exist_ok=True)
            
            # Generate and write the first set of problems
            problems = generate_x_by_y_problems(k, h, amount)
            with open(f"{n_dig_folder}/{k}_by_{h}_problems.txt", "w") as f:
                f.write("\n".join(problems))
            
            # Generate and write the second set of problems
            problems = generate_x_by_y_problems(k, h, amount, False)
            with open(f"{n_dig_folder}/{h}_by_{k}_problems.txt", "w") as g:
                g.write("\n".join(problems))


   