import re
import json
import string
from typing import List


def safe_prompt_format(prompt: str, variables: dict) -> str:
    """format promprt safely by extracting all placeholders from the prompt"""
    required_keys = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(prompt)
        if field_name
    }

    # Check for missing keys
    missing_keys = required_keys - variables.keys()
    if missing_keys:
        raise ValueError(f"Missing variables for prompt formatting: {missing_keys}")

    return prompt.format(**variables)


def extract_think_content(text):
    """
    Extracts the content between <think> and </think> tags in a given string.

    Args:
        text (str): The input string.

    Returns:
        str: The extracted content, or an empty string if not found.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1) if match else ""


def remove_think_content(text: str) -> str:
    # Remove <think>...</think> including the tags
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def extract_and_parse_json(text):
    """
    Extract JSON content from a string wrapped in ```json ... ``` and parse it.
    """
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[JSON Parse Error] {e}")
            return None
    else:
        print("[Warning] No JSON block found.")
        return None


def subtract_string(output: str, input_str: str) -> str:
    return output.replace(input_str, "")


#


def split_docs(text):
    # Splits by Score lines, keeping document blocks
    docs = re.split(r"(Score: \d+\.\d+)", text)
    paired_docs = []
    for i in range(1, len(docs), 2):
        score = docs[i].strip()
        content = docs[i + 1].strip()
        paired_docs.append(score + "\n" + content)
    return paired_docs


def extract_after_publisher(doc_text):
    # Extract content after Publisher: line
    match = re.search(r"(Publisher:.*)", doc_text, re.DOTALL)
    return doc_text[match.start() :].strip() if match else ""


def get_last_doc_number(docs):
    # Finds the highest "Document n. X" number
    last = 0
    for d in docs:
        match = re.search(r"Document n\.\s*(\d+)", d)
        if match:
            last = max(last, int(match.group(1)))
    return last


def renumber_documents(docs, start_index):
    # Replaces "Document n. X" with new sequential numbers
    new_docs = []
    current = start_index
    for doc in docs:
        new_doc = re.sub(r"Document n\.\s*\d+", f"Document n. {current}", doc, count=1)
        new_docs.append(new_doc)
        current += 1
    return new_docs


def filter_docs(docs1_text, docs2_text):
    docs1 = split_docs(docs1_text)
    docs2 = split_docs(docs2_text)

    # Unique signature from docs1 to compare with docs2
    seen_signatures = {extract_after_publisher(doc) for doc in docs1}

    # Filter docs2 to exclude duplicates based on Publisher content
    unique_docs2 = [
        doc for doc in docs2 if extract_after_publisher(doc) not in seen_signatures
    ]

    # Renumber docs2 starting after last doc number in docs1
    start_number = get_last_doc_number(docs1) + 1
    renumbered_docs2 = renumber_documents(unique_docs2, start_number)

    # Combine and return all docs as a single string
    combined = docs1 + renumbered_docs2
    return "\n\n".join(combined)


async def call_model(model, prompt: str) -> str:
    """Invoke an LLM model that may be a LangChain ChatModel (invoke) or a callable pipeline.

    Returns a plain string output.
    """
    try:
        result = await model.ainvoke(prompt)
        return getattr(result, "content", str(result))
    except Exception as e:
        raise Exception(f"Error calling model: {e}")


async def vector_db_retrieve_context(
    query: str, collection_ids: List[str], k: int = 5, score_threshold: float = 0.7
) -> str:
    """Retrieve similar documents from the vector DB and render as a docs string.

    Output format includes Score/Publisher/Content blocks to remain compatible with
    existing filtering utilities in this package.
    """
    # Deferred imports to avoid heavy deps at import time
    from src.core.vector_store_manager import VectorStoreManager

    manager = VectorStoreManager()

    try:
        results, _lat, _vec_err = await manager.retrieve_documents_with_latencies(
            collection_names=collection_ids,
            query=query,
            k=k,
            score_threshold=score_threshold,
        )

        blocks = []
        for idx, res in enumerate(results[:k], start=1):
            try:
                payload = getattr(res, "payload", {}) or {}
                text_val = payload.get("text") or payload.get("page_content") or ""
                publisher = payload.get("publisher") or payload.get("source") or ""
                score_val = getattr(res, "score", 0.0)

                lines = [
                    f"Score: {score_val:.4f}",
                    f"Document n. {idx}",
                    f"Publisher: {publisher}",
                    "Content:",
                    str(text_val or ""),
                ]
                blocks.append("\n".join(lines))
            except Exception:
                continue

        return "\n\n".join(blocks)
    except Exception:
        return ""
