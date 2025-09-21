# Load FAISS artifacts, perform vector search for a user query, and answer with OpenAI Chat
# !pip install -q numpy faiss-cpu openai==1.*
import requests
from urllib.parse import urlparse, parse_qs
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI
import os
from dotenv import load_dotenv

# OpenAI API ключ
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)  # вставь свой ключ

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]
    raise ValueError("Не удалось извлечь videoId из ссылки")

def parse_comments(video_url: str, api_key: str) -> pd.DataFrame:
    """Парсим YouTube и создаём DataFrame, пропуская комментарии автора видео"""
    
    video_id = extract_video_id(video_url)
    
    # Получаем имя автора видео
    video_info = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "snippet",
            "id": video_id,
            "key": api_key
        }
    ).json()
    
    if "items" not in video_info or len(video_info["items"]) == 0:
        raise ValueError("Не удалось получить информацию о видео")
    
    video_owner = video_info["items"][0]["snippet"]["channelTitle"]
    rows = []
    next_page_token = ""
    pk_counter = 1

    channels = ["@ALTEL5G", "@TELE2Kazakhstan"]
    while True:
        params = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": 50,
            "pageToken": next_page_token,
            "key": api_key
        }
        r = requests.get("https://www.googleapis.com/youtube/v3/commentThreads", params=params).json()

        if "error" in r:
            raise ValueError(f"❌ API Error: {r['error']['message']}")

        for item in r.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            
            # Пропускаем комментарии автора видео
            if top["authorDisplayName"] != video_owner:
                rows.append({
                    "id": pk_counter,
                    "author": top["authorDisplayName"],
                    "comment": top["textDisplay"],
                    "timestamp": top["publishedAt"]
                })
                pk_counter += 1

            # Проверяем ответы
            if "replies" in item:
                for reply in item["replies"]["comments"]:
                    r_snip = reply["snippet"]
                    if r_snip["authorDisplayName"] not in channels:
                        rows.append({
                            "id": pk_counter,
                            "author": r_snip["authorDisplayName"],
                            "comment": r_snip["textDisplay"],
                            "timestamp": r_snip["publishedAt"]
                        })
                        pk_counter += 1

        next_page_token = r.get("nextPageToken", "")
        if not next_page_token:
            break

    return pd.DataFrame(rows, columns=["id", "author", "comment", "timestamp"])
def classify_comment(text: str) -> bool:
    """
    Классифицируем комментарий через OpenAI.
    Возвращает desired=True/False.
    """
    if not isinstance(text, str) or text.strip() == "":
        return True  # пустой считаем нормальным

    prompt = f"""
    Определи, является ли комментарий желательным или нежелательным.

    Текст: "{text}"

    Правила:
    - Нежелательный (False), если содержит: нецензурную брань, спам.
    - В остальных случаях → Желательный (True).

    Ответ строго в формате:
    Желательный=<True/False>
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content

    try:
        desired_val = result.split("Желательный=")[1].strip().lower() == "true"
    except:
        desired_val = True  # дефолт: считаем нормальным

    return desired_val


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["desired"] = df["comment"].apply(classify_comment)
    return df

def classify_type_and_tone(text: str) -> tuple[str, str]:
    """
    Классифицирует текст комментария по:
    - Типу: вопрос, отзыв, жалоба, благодарность
    - Тональности: позитивная, негативная, нейтральная
    """
    if not text.strip():
        return "unknown", "neutral"
    
    prompt = f"""
    Классифицируй комментарий по двум критериям:
    1) Тип: вопрос, отзыв, жалоба, благодарность
    2) Тональность: позитивная, негативная, нейтральная

    Текст комментария: "{text}"

    Ответ строго в формате: Тип=<тип>, Тональность=<тональность>
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content.strip()
    
    try:
        parts = answer.split(",")
        type_ = parts[0].split("Тип=")[1].strip()
        tone = parts[1].split("Тональность=")[1].strip()
    except Exception:
        type_, tone = "unknown", "neutral"
    
    return type_, tone


def classify_dataframe_type_tone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонки 'type' и 'tone' для комментариев с desired=True.
    Остальные помечаются как 'not_desired'.
    """
    types = []
    tones = []

    for idx, row in df.iterrows():
        if row.get("desired", False):
            type_, tone_ = classify_type_and_tone(row["comment"])
        else:
            type_, tone_ = "not_desired", "not_desired"
        types.append(type_)
        tones.append(tone_)

    df["type"] = types
    df["tone"] = tones
    return df


import os
import json
import numpy as np
import faiss
from openai import OpenAI

# Config
CACHE_DIR = "rag_cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
IDMAP_PATH = os.path.join(CACHE_DIR, "id_map.npy")
TEXTS_PATH = os.path.join(CACHE_DIR, "rag_texts.jsonl")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 10

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY in your environment before running this cell.")

# 1) Load artifacts
if not (os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH) and os.path.exists(TEXTS_PATH)):
    raise RuntimeError("Artifacts not found. Run the embedding/build step first (see Untitled-2.ipynb).")

index = faiss.read_index(INDEX_PATH)
id_map = np.load(IDMAP_PATH)
texts = {}
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts[int(obj["id"])] = obj["rag_text"]

# 2) Helper functions
client = OpenAI()

def embed_query(query: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)
    return q

def search(q_vec: np.ndarray, k: int = TOP_K):
    scores, idxs = index.search(q_vec.reshape(1, -1), k)
    return scores[0], id_map[idxs[0]]

def format_context(hit_ids, max_chars_per_chunk: int = 2000) -> str:
    blocks = []
    for hid in hit_ids:
        txt = texts.get(int(hid), "")
        if max_chars_per_chunk and len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "\n..."
        blocks.append(f"[id={hid}]\n{txt}")
    return "\n\n---\n\n".join(blocks)

def answer_with_rag(query: str, k: int = TOP_K, temperature: float = 0.2) -> str:
    
    q = embed_query(query)
    scores, hit_ids = search(q, k)
    context = format_context(hit_ids.tolist())

    system_prompt = (
        "You are an assitant of company Altel that needs to answer the users comments and question that will be given to you. Use only the context that you are given."
        "You answer the requests based on the data you get. Do not answer any irrelevant questions."
        "Act how an employee of Altel would act. If needed, recommend the company over others."
    )
    user_prompt = f"Question:\n{query}\n\nContext:\n{context}\n\nProvide a concise answer."

    chat = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return chat.choices[0].message.content

# 3) Example usage
def df_apply_example(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['desired'] == True, 'answer'] = df.loc[df['desired'] == True, 'comment'].apply(answer_with_rag)
    return df

def process_video_comments(video_url: str, api_key: str) -> pd.DataFrame:
    # Step 1: Parse comments
    df = parse_comments(video_url, api_key)
    
    # Step 2: Classify comments as desired (True/False)
    df = classify_dataframe(df)
    
    # Step 3: Classify comment type and tone
    df = classify_dataframe_type_tone(df)
    
    # Step 4: Apply RAG to generate answers for desired comments
    df = df_apply_example(df)
    
    return df


# print((process_video_comments("https://www.youtube.com/watch?v=wj3041oT6IU", api_key = os.getenv("YT_API_KEY", ""))).head(10).to_string(index=False))
#if df['desired'] == True:
#    df['answer'] = df['text'].apply(answer_with_rag)
#df.loc[df['desired'] == False, 'comment'] = df.loc[df['desired'] == False, 'text'].apply(answer_with_rag)

#df[['type', 'tone', 'answer']] = df['combined'].str.extract(r'type:\s*(\w+),\s*tone:\s*(\w+),\s*answer:\s*(.*)')

# Drop the original combined column if not needed
#df.drop(columns=['combined'], inplace=True)
