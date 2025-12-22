try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sqlite3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from ultralytics import YOLO
from rembg import remove, new_session
from sentence_transformers import SentenceTransformer

import sqlite_vec
from sqlite_vec import serialize_float32

# -----------------------------
# Configuration Management
# -----------------------------
console = Console()
CONFIG_PATH = Path("philately.json")

DEFAULT_CONFIG = {
    "db_path": "philately.db",
    "stamps_dir": "stamps",
    "model_path": "model.pt",
    "clip_model_name": "clip-ViT-B-32",
    "vector_dimension": 512,
    "margin_percent": 0.02,
    "default_top": 5,
    "default_distance": 1.0,
    "yolo_conf": 0.4,
    "rembg_model": "u2netp"
}

def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=4))
        console.print(f"Created default config at {CONFIG_PATH}", style="yellow")
    
    # Merge defaults with local file to ensure all keys exist
    local_cfg = json.loads(CONFIG_PATH.read_text())
    return {**DEFAULT_CONFIG, **local_cfg}

# Global Config Object
CFG = load_config()

# Paths from Config
DB_PATH = Path(CFG["db_path"])
STAMPS_DIR = Path(CFG["stamps_dir"])
session = new_session(CFG["rembg_model"])

# -----------------------------
# Helpers
# -----------------------------
def banner(text):
    console.print(Panel(text, style="bold cyan"))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn

# -----------------------------
# Init
# -----------------------------
def init_project():
    if DB_PATH.exists():
        DB_PATH.unlink()
        console.print(f"Existing database {DB_PATH} deleted", style="yellow")

    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE stamps (
            id INTEGER PRIMARY KEY,
            album TEXT,
            page TEXT,
            stamp TEXT
        )
    """)

    conn.execute(f"""
        CREATE VIRTUAL TABLE stamp_vec USING vec0(
            embedding FLOAT[{CFG['vector_dimension']}] distance_metric=cosine
        )
    """)
    conn.commit()
    conn.close()

    STAMPS_DIR.mkdir(exist_ok=True)
    console.print("Project initialized with settings from JSON", style="bold green")

# -----------------------------
# Core Logic
# -----------------------------
def extract_stamps(album_folder, do_index=False, use_rembg=False):
    banner(f"Loading YOLO: {CFG['model_path']}")
    model = YOLO(CFG["model_path"])

    embedder = None
    conn = None
    cur = None

    if do_index:
        banner(f"Loading CLIP: {CFG['clip_model_name']}")
        embedder = SentenceTransformer(CFG["clip_model_name"])
        conn = get_db_connection()
        cur = conn.cursor()

    margin_pct = CFG["margin_percent"]
    images = list(Path(album_folder).glob("*.*"))

    for img_path in tqdm(images, desc="Processing Album"):
        results = model(str(img_path), conf=CFG["yolo_conf"], verbose=False)[0]

        if results.boxes is None or results.orig_img is None:
            continue

        img = results.orig_img
        h, w = img.shape[:2]

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            x1 = max(0, int(x1 - bw * margin_pct))
            y1 = max(0, int(y1 - bh * margin_pct))
            x2 = min(w, int(x2 + bw * margin_pct))
            y2 = min(h, int(y2 + bh * margin_pct))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            stamp = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")

            if use_rembg:
                # Use settings from config for rembg
                transparent = remove(stamp, session=session, alpha_matting=True)
                bbox = transparent.getbbox()
                final_img = transparent.crop(bbox) if bbox else transparent
            else:
                final_img = stamp

            name = f"{Path(album_folder).name}_{img_path.stem}_seg{i}.png"
            final_img.save(STAMPS_DIR / name)

            if do_index:
                cur.execute(
                    "INSERT INTO stamps(album,page,stamp) VALUES (?,?,?)",
                    (Path(album_folder).name, img_path.name, name),
                )
                stamp_id = cur.lastrowid
                vec = embedder.encode(final_img.convert("RGB")).astype(np.float32)
                cur.execute(
                    "INSERT INTO stamp_vec(rowid, embedding) VALUES (?,?)",
                    (stamp_id, serialize_float32(vec)),
                )

    if do_index:
        conn.commit()
        conn.close()
    
    console.print("Task complete", style="bold green")

def perform_search(query_type, query_val, top, distance):
    banner(f"{query_type.title()} Search")
    embedder = SentenceTransformer(CFG["clip_model_name"])
    
    if query_type == "image":
        query_input = Image.open(query_val).convert("RGB")
    else:
        query_input = query_val

    qvec = embedder.encode(query_input).astype(np.float32)
    conn = get_db_connection()
    
    rows = conn.execute(f"""
        SELECT stamps.album, stamps.page, stamps.stamp,
               vec_distance_cosine(stamp_vec.embedding, ?) AS distance
        FROM stamp_vec
        JOIN stamps ON stamp_vec.rowid = stamps.id
        WHERE vec_distance_cosine(stamp_vec.embedding, ?) <= ?
        ORDER BY distance ASC
        LIMIT ?
    """, (serialize_float32(qvec),serialize_float32(qvec), distance, top)).fetchall()
    
    conn.close()
    render_results(rows)

def render_results(rows):
    if not rows:
        return console.print("No results found", style="bold red")
    
    table = Table(title="Search Results")
    table.add_column("Album"); table.add_column("Page"); table.add_column("Stamp")
    table.add_column("Distance", justify="right")
    for r in rows:
        table.add_row(r[0], r[1], r[2], f"{r[3]:.4f}")
    console.print(table)

# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Philately Tool")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init")

    extract = sub.add_parser("extract")
    extract.add_argument("folder")
    extract.add_argument("--rembg", action="store_true", help="Remove background")

    index = sub.add_parser("index")
    index.add_argument("folder")
    index.add_argument("--rembg", action="store_true")

    s_img = sub.add_parser("search_image")
    s_img.add_argument("image")
    s_img.add_argument("--top", type=int, default=CFG["default_top"])
    s_img.add_argument("--distance", type=float, default=CFG["default_distance"])

    s_txt = sub.add_parser("search_text")
    s_txt.add_argument("text")
    s_txt.add_argument("--top", type=int, default=CFG["default_top"])
    s_txt.add_argument("--distance", type=float, default=CFG["default_distance"])

    args = parser.parse_args()

    if args.cmd == "init":
        init_project()
    elif args.cmd in ["extract", "index"]:
        extract_stamps(args.folder, do_index=(args.cmd == "index"), use_rembg=args.rembg)
    elif args.cmd == "search_image":
        perform_search("image", args.image, args.top, args.distance)
    elif args.cmd == "search_text":
        perform_search("text", args.text, args.top, args.distance)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()