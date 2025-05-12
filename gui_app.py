import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import re
from collections import Counter
from Levenshtein import distance as levenshtein_distance
import pandas as pd

# ==== Load model and tokenizer ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/University/NLP/Project/my_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
model.eval()

# ==== Load dataset and clean it ====
df = pd.read_csv(
    "D:/University/NLP/Project/arabic_dataset_classifiction/arabic_dataset_classifiction.csv"
)

# ==== Define all functions ====


def preprocess(sentence: str) -> str:
    sentence = sentence.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    sentence = re.sub(r"[^\u0600-\u06FF\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def data_vocab(dataframe, min_freq=3):
    words_freq = Counter()
    for text in dataframe["text"]:
        words_freq.update(text.split())
    return {word: freq for word, freq in words_freq.items() if freq >= min_freq}


def normalize_hamza(word: str) -> str:
    return (
        word.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ء", "")
    )


def find_misspellings(text: str, vocab: dict, threshold: float = 0.28) -> list:
    words = text.split()
    misspelled_indices = []
    for i, word in enumerate(words):
        if word not in vocab and normalize_hamza(word) not in vocab:
            masked_words = words.copy()
            masked_words[i] = tokenizer.mask_token
            masked_sentence = " ".join(masked_words)
            inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
            mask_token_index = torch.where(
                inputs["input_ids"] == tokenizer.mask_token_id
            )[1]
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, mask_token_index]
                probs = torch.softmax(logits, dim=-1).squeeze()
                word_id = tokenizer.encode(word, add_special_tokens=False)
                word_prob = torch.mean(probs[word_id]) if word_id else 0
            if word_prob < threshold:
                misspelled_indices.append(i)
    return misspelled_indices


def generate_masked_sentences(text: str, misspelled_indices: list) -> list:
    words = text.split()
    return [
        " ".join(words[:idx] + [tokenizer.mask_token] + words[idx + 1 :])
        for idx in misspelled_indices
    ]


def predict(masked_sentence: str, top_k=25) -> list:
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, mask_token_index]
    probs = torch.softmax(logits, dim=-1).squeeze()
    top_k_tokens = torch.topk(probs, top_k)
    predictions = []
    for token_id in top_k_tokens.indices:
        token = tokenizer.decode([token_id]).strip()
        if re.match(r"^[\u0600-\u06FF]{2,}$", token):
            predictions.append(token)
    return predictions


def pipeline(input_text: str, vocab: dict, verbose: bool = True) -> str:
    processed_text = preprocess(input_text)
    vocab = data_vocab(df, min_freq=3)
    misspelled_indices = find_misspellings(processed_text, vocab)

    if not misspelled_indices:
        if verbose:
            print("✅ لا توجد أخطاء إملائية واضحة.")
        return processed_text, {}

    masked_sentences = generate_masked_sentences(processed_text, misspelled_indices)
    words = processed_text.split()
    corrections = {}

    for idx, masked in zip(misspelled_indices, masked_sentences):
        original_word = words[idx]
        candidates = predict(masked)
        if candidates:
            best_candidate = min(
                candidates, key=lambda c: levenshtein_distance(c, original_word)
            )
            corrections[original_word] = best_candidate
            words[idx] = best_candidate

    corrected_sentence = " ".join(words)

    if verbose:
        print("🔍 الكلمات التي تم تصحيحها:")
        for original, corrected in corrections.items():
            print(f" - {original} ➤ {corrected}")

    return corrected_sentence, corrections


# ==== Clean dataset ====
df = df.drop(columns=["targe"], errors="ignore").dropna().drop_duplicates()
df["text"] = df["text"].apply(preprocess)
df["text"] = df["text"].apply(lambda x: x if len(x.split()) > 5 else None)
df = df.dropna().reset_index(drop=True)

# ==== Build vocab once globally ====
words_freq = data_vocab(df)


# ==== Build GUI ====
def correct_text():
    input_text = entry.get("1.0", tk.END).strip()
    if not input_text.strip():
        result_var.set("❌ الرجاء إدخال نص أولاً.")
        return

    corrected, corrections = pipeline(input_text, words_freq)

    if corrected.strip() == preprocess(input_text).strip():
        result_var.set(f"✅ لا توجد أخطاء إملائية:\n\n{corrected}")
    else:
        # عرض الكلمات المصححة
        corrections_text = "🔍 الكلمات التي تم تصحيحها:\n"
        for original, corrected_word in corrections.items():
            corrections_text += f" - {original} ➤ {corrected_word}\n"

        result_var.set(f"❌ قبل التصحيح:\n{input_text}\n\n✅ بعد التصحيح:\n{corrected}\n\n{corrections_text}")

    entry.delete("1.0", tk.END)


root = tk.Tk()
root.title("AutoCorrect On Arabic")
root.geometry("800x600")  # Set window size
root.config(bg="#f4f4f4")  # Background color

# Use a frame for the input section
input_frame = tk.Frame(root, bg="#f4f4f4")
input_frame.pack(pady=20)

tk.Label(
    input_frame, text="أدخل الجملة:", font=("Arial", 14, "bold"), bg="#f4f4f4"
).pack()
# Using Text widget for multiline input
entry = tk.Text(
    input_frame,
    width=40,
    height=5,
    font=("Arial", 14),
    bd=2,
    relief="solid",
    wrap="word",
    padx=10,
    pady=10,
)
entry.pack(pady=10)
entry.tag_configure("right", justify="right")
entry.insert("1.0", "")
entry.tag_add("right", "1.0", "end")

# Button with custom styling
tk.Button(
    input_frame,
    text="تصحيح",
    font=("Arial", 14, "bold"),
    bg="#5DBCFC",
    fg="white",
    activebackground="#0073e6",
    activeforeground="white",
    width=20,
    height=2,
    command=correct_text,
).pack()

# Output frame for results
output_frame = tk.Frame(root, bg="#f4f4f4")
output_frame.pack(pady=20)

tk.Label(
    output_frame, text="نتيجة التصحيح:", font=("Arial", 14, "bold"), bg="#f4f4f4"
).pack()
result_var = tk.StringVar()
tk.Label(
    output_frame,
    textvariable=result_var,
    wraplength=550,
    justify="right",
    font=("Arial", 12),
    bg="#f4f4f4",
).pack()


root.mainloop()
