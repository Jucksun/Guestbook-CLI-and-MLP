# Guestbook-CLI-and-MLP
I made this for my Grad Party! The full program logs messages to a Database AND spots out a guess, but this one just guesses!

# Guestbook: PyTorch iMessage Speaker Classifier

A machine learning project and CLI tool that predicts the sender of an iMessage based on feature engineering of text styles (capitalization ratios, punctuation counts, and message lengths). I built it using PyTorch and manual z-score feature normalization!
(Also just know, the prédictions are based on my top contacts, and THUS you won't know who you're similar to by name)

## Features

- Custom PyTorch Architecture: multi layer perceptron (MLP) expérimentation trained to classify distinct message signatures :p.
- Manual Feature Scaling: z-score normalization ($z = \frac{x - \mu}{\sigma}$)(thank you markdown formula library) implemented during training(and inference) to eliminate length bias artifacts. I didn't want to rely on external wrapper,s because I wanted to learn the underlying processes.
- Interactive Terminal CLI: inference terminal featuring softmax confidence distribution outputs for each speaker prediction. Pretty fun to make tbh, and I Learned a lot from this surprisingly.

## Architecture & Data!!

1. Feature Engineering (also pretty fun): Extracts caps_ratio, punctuation_count, and message_length from raw text input. I'm planning to fix the 4th feature, which is currently just 0.0, for transparency.
2. Normalization: Input vectors are scaled using the training dataset's mean and standard deviation parameters.
3.  Structure of model: 3 layer neural network (Linear -> ReLU -> Linear -> ReLU -> Linear, which... I also learned about from this! ReLU is cool).

## Project Structure

- CLI.py: interface for live predictions!
- train.py: Data, normalization, model training, AND evaluation script. Phew.
- model.py: PyTorch DetermineSpeaker model definition (I tihnk there were 3 version? I uploaded the most current).
- preprocessing.py: Feature extraction and string cleaning.
- model_weights_famille_2.pth: Saved PyTorch state dictionary. I also uploaded the original weights if you want to compare before and after the z-score update!

1. Clone and install required packages!
2. Launch the CLI tool to test input predictions live:
\`\`\`bash
python3 CLI.py
\`\`\`

### To-Do!
- add 4th feature
- fix so clones can refrain?


ANYWAYS bye thanks : ) this is my first real readme








## Performance

- **Test Accuracy:** ~88.8%
- **Evaluation Metric:** Cross-Entropy Loss with Adam optimizer
