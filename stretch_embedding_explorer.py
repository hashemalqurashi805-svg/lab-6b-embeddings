import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import os

# 1. إعدادات التصوير البياني
plt.rcParams['figure.figsize'] = [12, 8]

# 2. فئات الكلمات (GloVe Selection) - 5 فئات كما هو مطلوب في القيود
word_categories = {
    "Technology": ["software", "hardware", "internet", "algorithm", "robot", "digital", "network", "server", "coding", "data"],
    "Sports": ["football", "basketball", "tennis", "athlete", "stadium", "referee", "champion", "olympics", "tournament", "coach"],
    "Countries": ["jordan", "saudi", "egypt", "palestine", "iraq", "lebanon", "kuwait", "qatar", "oman", "syria"],
    "Emotions": ["happy", "sad", "angry", "excited", "fear", "joy", "grief", "brave", "anxious", "calm"],
    "Business": ["economy", "market", "finance", "company", "investment", "profit", "revenue", "bank", "trade", "startup"]
}

def get_dummy_glove_embeddings(word_dict):
    words = []
    embeddings = []
    labels = []
    for cat, w_list in word_dict.items():
        for w in w_list:
            words.append(w)
            embeddings.append(np.random.rand(50)) # محاكاة لمتجهات GloVe 50d
            labels.append(cat)
    return np.array(embeddings), words, labels

# 3. معالجة المقالات باستخدام DistilBERT
def get_distilbert_embeddings(texts):
    print("Loading DistilBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # استخدام الـ CLS token لتمثيل المقالة دلالياً
    return outputs.last_hidden_state[:, 0, :].numpy()

# 4. دالة تقليل الأبعاد المعدلة (تجنب خطأ Perplexity)
def reduce_and_plot(embeddings, labels, title, filename, annotations=None):
    print(f"Generating plot: {title}...")
    
    n_samples = embeddings.shape[0]
    # تعديل قيمة perplexity لتكون دائماً أقل من عدد العينات (n_samples)
    safe_perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42, init='pca')
    reduced_data = tsne.fit_transform(embeddings)
    
    plt.figure()
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=label, color=color, s=100)
    
    if annotations:
        for i, text in enumerate(annotations):
            # إضافة تسمية توضيحية بأول 30 حرفاً كما في صورة image_e5b53c.png
            plt.annotate(text[:30], (reduced_data[i, 0], reduced_data[i, 1]), alpha=0.7, fontsize=8)
            
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # حفظ المخطط كصورة لمعاينته في VS Code
    plt.savefig(filename)
    print(f"Plot saved successfully as: {filename}")
    plt.show()

# 5. التشغيل الرئيسي
def main():
    # أ- الجزء الأول: كلمات GloVe
    print("--- Step 1: Word Embeddings ---")
    g_embeddings, words, g_labels = get_dummy_glove_embeddings(word_categories)
    reduce_and_plot(g_embeddings, g_labels, "GloVe Word Embeddings Clusters", "glove_clusters.png")

    # ب- الجزء الثاني: مقالات BBC News
    csv_path = 'data/bbc_news.csv'
    if os.path.exists(csv_path):
        print("\n--- Step 2: Document Embeddings ---")
        df = pd.read_csv(csv_path)
        # اختيار 20 مقالة (4 من كل فئة) لتحقيق شرط القيود
        sampled_df = df.groupby('category').head(4).copy()
        
        doc_embeddings = get_distilbert_embeddings(sampled_df['text'].tolist())
        reduce_and_plot(doc_embeddings, sampled_df['category'].tolist(), 
                       "DistilBERT Document Embeddings", 
                       "bbc_document_clusters.png",
                       annotations=sampled_df['title'].tolist())
    else:
        print(f"\n[!] Error: {csv_path} not found. Please check the 'data' folder.")

if __name__ == "__main__":
    main()