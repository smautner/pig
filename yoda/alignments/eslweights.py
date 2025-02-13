import re

def getWeights(filepath):
    """
    Parses an Rfam seed file (Stockholm format) annotated with ESL weights
    (specifically, the GS and WT annotations) and extracts the weight vector
    for each family.

    Args:
        filepath: Path to the Rfam seed file.

    Returns:
        A dictionary where keys are Rfam family names (accessions) and values
        are lists of floats representing the weight vector (from the WT line).
        Returns an empty dictionary if no families or weights are found.  Handles
        multiple Stockholm alignments within the same file.
    """

    family_weights = {}
    current_family = None
    weights = None

    DEBUG = True
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Match family accession line (e.g., #=GF AC   RF00001)
            match_ac = re.match(r'^#=GF\s+AC\s+(\S+)', line)
            if match_ac:
                # PREPARE TO READ
                current_family = match_ac.group(1)
                weights = []  # Reset weights for the new family
                if DEBUG: print('new family:',current_family)

            # Match the weight line (e.g., #=GS WT   0.5;0.7;...;0.2)
            match_wt = re.match(r'^#=GS\s+.+\s+WT\s+(.+)', line)
            if match_wt:
                weight_string = match_wt.group(1)
                # Split by semicolons and convert to floats
                weights.append(float(weight_string))
                if DEBUG: print('new weight:',weight_string)

            # Check end of stockholm alignment using '//'
            if line == '//':
                if current_family and weights:
                    family_weights[current_family] = weights


    return family_weights


def main():
    """
    Example usage.  Replace 'your_rfam_seed_file.sto' with the actual path
    to your file.
    """
    filepath = 'weights.rfam'
    family_weight_dict = getWeights(filepath)

    if family_weight_dict:
        print("Extracted Family Weights:")
        for family, weights in family_weight_dict.items():
            print(f"  {family}: {weights}")
    else:
        print("No family weights found or an error occurred.")


if __name__ == "__main__":
    main()
