import torch
import string
import re
import sys
from pathlib import Path
import emoji

from model import DetermineSpeaker

MEANS = [0.470778, 0.172344, 1.146874, 25.819583]
STDS = [0.499188, 0.316679, 4.604921, 37.002466]

SPEAKER_MAP = {
    0: "Me",
    1: "Person 1",
    2: "Person 2",
    3: "Person 3",
    4: "Person 4",
    5: "Person 5"
}




def extract_input_features(text: str):
    """Transforms raw user input into the 4 feature tensor """
    text_clean = emoji.demojize(text)
    text_clean = re.sub(r"[^\w\s" + re.escape(string.punctuation) + r"]", "", text_clean)
    
    caps_ratio = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if text_clean else 0.0
    punct_count = sum(1 for c in text_clean if c in string.punctuation)
    msg_len = len(text_clean)

    norm_caps = (caps_ratio - MEANS[0]) / STDS[0]
    norm_punct = (punct_count - MEANS[1]) / STDS[1]
    norm_len = (msg_len - MEANS[2]) / STDS[2]
    norm_dummy = (0.0 - MEANS[3]) / STDS[3] if STDS[3] != 0 else 0.0

#This is a fixed versionn because before I didn't make it fit to the new weights that I scaled with z-score
    features = [norm_caps, norm_punct, norm_len, norm_dummy]
    return torch.tensor([features], dtype=torch.float32)
   
    

def main():
    
    model = DetermineSpeaker()

    MODEL_PATH = Path(__file__).parent / "model_weights_famille_2.pth"

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    
    
    model.eval()

    print("--- Guestbook Speaker Inference Terminal ---")
    print("Type a message to predict the speaker (or 'exit' to quit):\n")

    while True:
        user_input = input("> ")
        if user_input.lower().strip() in ["exit", "quit"]:
            break
            
        if not user_input.strip():
            continue

        x_in = extract_input_features(user_input)
        
        with torch.no_grad():
            logits = model(x_in)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        print(f"\nPredicted Speaker: {SPEAKER_MAP.get(predicted_class, 'Unknown')}")
        print("Confidence Distribution:")
        for idx, prob in enumerate(probabilities):
            print(f"  {SPEAKER_MAP.get(idx, f'Speaker {idx}')}: {prob.item() * 100:.1f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()
