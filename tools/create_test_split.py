import json
import random

def split_test_validate(input_file, train_file, test_file, validate_file):
    """
    Splits the data from input_file into test and validate lists based on a specified probability
    """
    # Load the data from the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    train = []
    test = []
    validate = []

    modes = ["train", "test", "validate"]
    weights = [0.7, 0.2, 0.1]

    # Split the data
    for item in data:
        mode = random.choices(modes, weights)[0]
        if mode == "train":
            train.append(item)
        elif mode == "test":
            test.append(item)
        else:
            validate.append(item)

    # Write the train list to the train file
    with open(train_file, 'w') as f:
        json.dump(train, f, indent=4)

    # Write the test list to the test file
    with open(test_file, 'w') as f:
        json.dump(test, f, indent=4)

    # Write the validate list to the validate file
    with open(validate_file, 'w') as f:
        json.dump(validate, f, indent=4)

# Example usage
if __name__ == "__main__":
    split_test_validate(
        input_file="augmented_test_set.json",
        train_file="train_set.json",
        test_file="test_set.json",
        validate_file="validate_set.json"
    )