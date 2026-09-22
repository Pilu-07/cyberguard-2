"""
Produce benign_decoded.csv and malicious_decoded.csv.

- benign_decoded.csv  : parsed from existing qr_decoded_urls.csv (benign rows only)
- malicious_decoded.csv : decoded fresh from e:/project 2/malicious/*.png
  using the same multi-engine pipeline as run_qr_pipeline.py

Output columns (same as qr_decoded_urls.csv):
  qr_id, image_filename, source_folder, label, decoded_content,
  normalized_url, domain, domain_length, path_length, query_length,
  fragment_length, subdomain_count, has_https, has_http, has_ip_address,
  has_port, has_at_symbol, has_encoded_characters, has_shortener_pattern,
  digit_count, special_character_count
"""

import os, re, sys, csv, ipaddress
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count

import cv2
import zxingcpp
import pyzbar.pyzbar as pyzbar

# ── Config ────────────────────────────────────────────────────────────────────
BENIGN_RAW   = r"e:\project 2\qr_decoding_output\qr_decoded_urls.csv"
MAL_DIR      = r"e:\project 2\malicious"
OUT_DIR      = r"e:\project 2\qr_decoding_output"
BENIGN_OUT   = os.path.join(OUT_DIR, "benign_decoded.csv")
MAL_OUT      = os.path.join(OUT_DIR, "malicious_decoded.csv")

COLS = [
    "qr_id","image_filename","source_folder","label","decoded_content",
    "normalized_url","domain","domain_length","path_length","query_length",
    "fragment_length","subdomain_count","has_https","has_http","has_ip_address",
    "has_port","has_at_symbol","has_encoded_characters","has_shortener_pattern",
    "digit_count","special_character_count"
]

KNOWN_SHORTENERS = {
    "bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly",
    "adf.ly","bit.do","rebrand.ly","cutt.ly","shorturl.at","tiny.cc",
    "lnkd.in","db.tt","qr.ae","trib.al","bl.ink","cur.lv","bc.vc",
}

SERIES_RE = re.compile(r"^\s*\d+\s+(https?://\S+)", re.IGNORECASE | re.MULTILINE)
GENERIC_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)


# ── Part A: Fix benign_decoded.csv from existing raw CSV ──────────────────────
def clean_decoded(val):
    if not val or str(val).strip() == "nan":
        return val
    val = re.sub(r"\nName:.*$", "", str(val), flags=re.DOTALL).strip()
    val = re.sub(r"^\d+\s+", "", val).strip()
    return val

def build_benign():
    print("Building benign_decoded.csv from existing raw CSV ...")
    # Use csv module directly to handle multi-line quoted fields correctly
    rows = []
    with open(BENIGN_RAW, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["decoded_content"] = clean_decoded(row.get("decoded_content", ""))
            # Keep only the columns we want
            rows.append({c: row.get(c, "") for c in COLS})

    with open(BENIGN_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Benign rows saved : {len(rows):,}  ->  {BENIGN_OUT}")
    return rows


# ── QR decoding helpers (mirrors run_qr_pipeline.py) ─────────────────────────
def decode_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    for attempt in [img]:
        try:
            results = zxingcpp.read_barcodes(attempt)
            for r in results:
                if r.text:
                    return r.text
        except Exception:
            pass
    try:
        det = cv2.QRCodeDetector()
        text, _, _ = det.detectAndDecode(img)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        for b in pyzbar.decode(img):
            if b.data:
                return b.data.decode("utf-8", errors="replace")
    except Exception:
        pass
    # Preprocessing fallback
    try:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = zxingcpp.read_barcodes(thresh)
        if res and res[0].text:
            return res[0].text
    except Exception:
        pass
    return None


def extract_normalize(decoded_text):
    if not decoded_text:
        return None
    raw = decoded_text.strip()
    m = SERIES_RE.search(raw)
    if m:
        url = m.group(1).strip()
    elif raw.lower().startswith(("http://", "https://")):
        url = raw.split()[0]
    else:
        m2 = GENERIC_RE.search(raw)
        if not m2:
            return None
        url = m2.group(1).strip()
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return None
        norm = f"{p.scheme.lower()}://{p.netloc.lower()}{p.path}"
        if p.query:
            norm += f"?{p.query}"
        if p.fragment:
            norm += f"#{p.fragment}"
        return norm
    except Exception:
        return None


def extract_features(norm_url):
    f = dict(domain="", domain_length=0, path_length=0, query_length=0,
             fragment_length=0, subdomain_count=0, has_https=0, has_http=0,
             has_ip_address=0, has_port=0, has_at_symbol=0,
             has_encoded_characters=0, has_shortener_pattern=0,
             digit_count=0, special_character_count=0)
    if not norm_url:
        return f
    f["digit_count"]           = sum(1 for c in norm_url if c.isdigit())
    f["special_character_count"] = sum(1 for c in norm_url if not c.isalnum())
    f["has_at_symbol"]         = 1 if "@" in norm_url else 0
    f["has_encoded_characters"] = 1 if "%" in norm_url else 0
    try:
        p = urlparse(norm_url)
        f["has_https"]      = 1 if p.scheme == "https" else 0
        f["has_http"]       = 1 if p.scheme == "http"  else 0
        host = p.hostname or ""
        f["domain"]         = host
        f["domain_length"]  = len(host)
        f["path_length"]    = len(p.path)
        f["query_length"]   = len(p.query)
        f["fragment_length"] = len(p.fragment)
        f["has_port"]       = 1 if p.port else 0
        parts = host.split(".")
        f["subdomain_count"] = max(len(parts) - 2, 0) if len(parts) > 2 else 0
        try:
            ipaddress.ip_address(host)
            f["has_ip_address"] = 1
        except ValueError:
            pass
        f["has_shortener_pattern"] = 1 if host in KNOWN_SHORTENERS else 0
    except Exception:
        pass
    return f


def process_one(args):
    idx, img_path, fname = args
    decoded = decode_image(img_path)
    norm    = extract_normalize(decoded)
    feats   = extract_features(norm)
    return dict(
        qr_id            = f"MAL_{idx:06d}",
        image_filename   = fname,
        source_folder    = "malicious",
        label            = 1,
        decoded_content  = decoded or "",
        normalized_url   = norm or "",
        **feats,
    )


# ── Part B: Decode malicious images ──────────────────────────────────────────
def build_malicious():
    print("Building malicious_decoded.csv from malicious QR images ...")
    files = sorted([f for f in os.listdir(MAL_DIR) if f.lower().endswith(".png")])
    total = len(files)
    print(f"  Found {total:,} malicious images")

    args = [(i+1, os.path.join(MAL_DIR, f), f) for i, f in enumerate(files)]

    workers = max(1, cpu_count() - 1)
    print(f"  Using {workers} workers ...")

    rows = []
    with Pool(workers) as pool:
        for done, row in enumerate(pool.imap(process_one, args, chunksize=50), 1):
            rows.append({c: row.get(c, "") for c in COLS})
            if done % 10000 == 0 or done == total:
                print(f"  [{done:,}/{total:,}]  decoded={sum(1 for r in rows if r['decoded_content'])}")

    with open(MAL_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)

    decoded_count = sum(1 for r in rows if r["decoded_content"])
    url_count     = sum(1 for r in rows if r["normalized_url"])
    print(f"  Malicious rows saved : {len(rows):,}  (decoded: {decoded_count:,}, URLs: {url_count:,})")
    print(f"  -> {MAL_OUT}")


if __name__ == "__main__":
    build_benign()
    build_malicious()
    print("\nDone! Both CSV files are ready.")
