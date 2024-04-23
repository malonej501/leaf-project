input = "../Naturalis_ang_species_occurrence.csv"


def row_count(input):
    with open(input) as f:
        for i, l in enumerate(f):
            pass
    return i


print(row_count(input))
