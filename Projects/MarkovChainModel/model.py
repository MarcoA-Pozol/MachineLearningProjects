import markovify
from datasets import sample1, sample2

model = markovify.Text((sample1, sample2), state_size=2)

# Using model to generate text
for _ in range(2): # Generate 10 sentences
    print(model.make_sentence())