import markovify # Model for text generation
from datasets import sample1, sample2, sample3, sample4 # Datasets to train the model

model = markovify.Text((sample1, sample2), state_size=1)

# Using model to generate text
for _ in range(2): # Generate 10 sentences
    print(model.make_sentence())

def save_model(model:object):
    """
        Save the model as json.
    """
    with open("markov_model.json", "w") as file:
        file.write(model.to_json())


def main():
    # Save model
    save_model(model)

    # Obtain saved model
    with open("markov_model.json", "r") as file:
        existing_model = markovify.Text.from_json(file.read())

    # Train new model with extra data
    new_model = markovify.Text((sample3, sample4), state_size=1)

    # Combine old and new model
    combined_model = markovify.combine([existing_model, new_model], [1, 1])

    # Generate text from improved model
    for _ in range(3):
        print(combined_model.to_json())

    # Save updated model
    with open("markov_model.json", "w") as f:
        f.write(combined_model.to_json())


if __name__ == "__main__":
    main()
