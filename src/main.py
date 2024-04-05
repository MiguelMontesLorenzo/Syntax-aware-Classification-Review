from src.data_processing_parsing import load_data
from src.grammar_conversion import PCFG
from src.parsing import CKY
import os


def main() -> None:
    """
    Args:
    - None

    Returns:
    - None
    """
    # Path to the directory containing the files
    path: str = "data_parsing"

    # File name to save the grammar
    filename: str = "grammar_rules.txt"
    filepath: str = path + "/" + filename

    # Load grammar if it doesnt exist
    if not os.path.exists(filepath):
        print("Creating file...")
        load_data(path=path, filepath=filepath)

    # Probability Context Free Grammar
    print("Creating PCFG...")
    pcfg: PCFG = PCFG()
    pcfg.read_rules(filepath)
    pcfg.compute_probabilities()

    # Parsing
    cky: CKY = CKY(pcfg, "my first lover was on TV")
    print(cky.parse())


if __name__ == "__main__":
    main()
