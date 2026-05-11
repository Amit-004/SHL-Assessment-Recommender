import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load catalog
with open("app/data/catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)


def safe_join(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value or "")


# Create searchable text for each assessment
documents = []

for item in catalog:
    text = f"""
    Assessment Name: {item.get('name', '')}

    Description:
    {item.get('description', '')}

    Job Levels:
    {safe_join(item.get('job_levels', []))}

    Categories:
    {safe_join(item.get('keys', []))}

    Languages:
    {safe_join(item.get('languages', []))}

    Duration:
    {item.get('duration', '')}

    Remote Testing:
    {item.get('remote', '')}

    Adaptive:
    {item.get('adaptive', '')}

    Search Keywords:
    {item.get('name', '')}
    {item.get('description', '')}
    software developer engineer programmer coding java python sql javascript frontend backend full stack
    communication business personality behavior aptitude cognitive leadership management sales customer service
    data science machine learning ai cloud aws testing selenium automation
    """
    documents.append(text)


# Create embeddings
embeddings = model.encode(documents, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")


# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def search_assessments(query, top_k=10):

    query_lower = query.lower()

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, 30)

    scored_results = []

    for idx in indices[0]:

        item = catalog[idx]

        score = 0

        name = item.get("name", "").lower()
        desc = item.get("description", "").lower()
        keys = " ".join(item.get("keys", [])).lower()

        combined = f"{name} {desc} {keys}"

        # --------------------------------
        # Keyword boosting
        # --------------------------------

        keywords = query_lower.split()

        for word in keywords:
            if word in combined:
                score += 2

        # Strong boosts
        if "java" in query_lower and "java" in combined:
            score += 10

        if "communication" in query_lower and "communication" in combined:
            score += 8

        if "personality" in query_lower and "personality" in keys:
            score += 10

        if "developer" in query_lower and (
            "coding" in combined or
            "programming" in combined
        ):
            score += 6

        if "aptitude" in query_lower and "aptitude" in keys:
            score += 10

        scored_results.append((score, item))

    # Sort by custom score
    scored_results.sort(key=lambda x: x[0], reverse=True)

    final_results = []

    seen = set()

    for score, item in scored_results:

        if item["name"] in seen:
            continue

        seen.add(item["name"])

        final_results.append({
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "description": item.get("description", ""),
            "test_type": ", ".join(item.get("keys", [])),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
            "job_levels": item.get("job_levels", [])
        })

        if len(final_results) >= top_k:
            break

    return final_results