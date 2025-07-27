from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def read_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

def detect_plagiarism(input_text, source_docs):
    input_embedding = model.encode([input_text])
    scores = {}
    for name, content in source_docs.items():
        doc_embedding = model.encode([content])
        similarity = cosine_similarity(input_embedding, doc_embedding)[0][0]
        scores[name] = round(similarity * 100, 2)
    return scores

input_file = 'C:/Users/Lakshminarasimha/slashmark/basic/submission.txt'
source_folder = 'C:/Users/Lakshminarasimha/slashmark/basic/sample_dataset'

with open(input_file, 'r', encoding='utf-8') as f:
    input_text = f.read()

source_documents = read_documents(source_folder)
results = detect_plagiarism(input_text, source_documents)

print("\nPlagiarism Report:")
for doc, score in results.items():
    if score > 60:
        print(f"{doc}: {score}% similar")
    else:
        print(f"{doc}: {score}% similar")
