from datasets import load_dataset
import json

def main():
    # Load the MedDialog English dataset (first 10 conversations)
    dataset = load_dataset("medical_dialog", "en", split="train[:10]")

    conversations = []
    for entry in dataset:
        conversations.append({
            "goal": entry["goal"],
            "dialogue": entry["dialogue"]
        })

    # Print a sample conversation
    print("Sample conversation:")
    for turn in conversations[0]["dialogue"]:
        print(turn)

    # Save to JSON
    with open("meddialog_sample.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print("Saved 10 sample conversations to meddialog_sample.json")

if __name__ == "__main__":
    main()