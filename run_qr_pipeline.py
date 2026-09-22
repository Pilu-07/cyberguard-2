import os
import sys
import re
import csv
import time
import ipaddress
from urllib.parse import urlparse, unquote
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict

import cv2
import zxingcpp
import pyzbar.pyzbar as pyzbar

# Known common URL shorteners
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "trib.al", "bl.ink", "cur.lv", "bc.vc",
    "po.st", "ity.im", "q-r.to", "u.to", "v.gd", "x.co", "soo.gd"
}

# Regex to detect pandas series representation
SERIES_URL_REGEX = re.compile(r"^\s*\d+\s+(https?://[^\s\r\n]+)", re.IGNORECASE | re.MULTILINE)
GENERIC_URL_REGEX = re.compile(r"(https?://[^\s\r\n]+)", re.IGNORECASE)

def decode_image_qr(image_path):
    """
    Multi-engine QR decoder cascade:
    1. OpenCV QRCodeDetector
    2. zxing-cpp
    3. pyzbar
    4. Image pre-processing (contrast / thresholding) if needed.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "corrupt_or_unreadable_image"

    # 1. Try zxing-cpp directly (fastest and highly robust for 1D/2D)
    try:
        results = zxingcpp.read_barcodes(img)
        for r in results:
            if r.format == zxingcpp.BarcodeFormat.QRCode and r.text:
                return r.text, None
            elif r.text:  # fallback if format is unspecified
                return r.text, None
    except Exception as e:
        pass

    # 2. Try OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        decoded_text, points, _ = detector.detectAndDecode(img)
        if decoded_text and decoded_text.strip():
            return decoded_text.strip(), None
    except Exception as e:
        pass

    # 3. Try pyzbar
    try:
        barcodes = pyzbar.decode(img)
        for b in barcodes:
            if b.data:
                return b.data.decode("utf-8", errors="replace"), None
    except Exception as e:
        pass

    # 4. Fallback: Image preprocessing (Otsu thresholding and CLAHE)
    try:
        # Otsu thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = zxingcpp.read_barcodes(thresh)
        if res and res[0].text:
            return res[0].text, None

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_img = clahe.apply(img)
        res = zxingcpp.read_barcodes(cl_img)
        if res and res[0].text:
            return res[0].text, None
    except Exception:
        pass

    return None, "no_qr_detected_or_unreadable"

def extract_and_normalize_url(decoded_text):
    """
    Determines content type, extracts URL if present, and normalizes safely.
    NEVER visits the URL.
    """
    if not decoded_text:
        return None, None, "unknown"

    raw_text = decoded_text.strip()

    # Check for pandas Series pattern
    series_match = SERIES_URL_REGEX.search(raw_text)
    if series_match:
        extracted = series_match.group(1).strip()
    elif raw_text.lower().startswith(("http://", "https://")):
        extracted = raw_text.split()[0]
    else:
        # Check for generic URL inside text
        generic_match = GENERIC_URL_REGEX.search(raw_text)
        if generic_match:
            extracted = generic_match.group(1).strip()
        else:
            # Not a URL
            return None, raw_text, "non_url"

    # Normalize extracted URL
    try:
        # Remove trailing periods if it was part of pandas '...' or sentence
        # but preserve path query
        cleaned_url = extracted.strip()
        parsed = urlparse(cleaned_url)
        if not parsed.scheme or not parsed.netloc:
            return None, raw_text, "non_url"

        # Safe normalization: lowercase scheme and host/domain
        norm_scheme = parsed.scheme.lower()
        norm_netloc = parsed.netloc.lower()
        
        # Reconstruct normalized URL
        norm_url = f"{norm_scheme}://{norm_netloc}{parsed.path}"
        if parsed.query:
            norm_url += f"?{parsed.query}"
        if parsed.fragment:
            norm_url += f"#{parsed.fragment}"

        return norm_url, extracted, "url"
    except Exception:
        return None, raw_text, "non_url"

def extract_url_features(normalized_url):
    """
    Deterministic URL feature extraction without visiting the URL.
    """
    features = {
        "url_length": 0,
        "domain": "",
        "domain_length": 0,
        "path_length": 0,
        "query_length": 0,
        "fragment_length": 0,
        "subdomain_count": 0,
        "has_https": 0,
        "has_http": 0,
        "has_ip_address": 0,
        "has_port": 0,
        "has_at_symbol": 0,
        "has_encoded_characters": 0,
        "has_shortener_pattern": 0,
        "digit_count": 0,
        "special_character_count": 0,
    }

    if not normalized_url:
        return features

    features["url_length"] = len(normalized_url)
    features["digit_count"] = sum(1 for c in normalized_url if c.isdigit())
    
    # Special characters (anything not alphanumeric)
    features["special_character_count"] = sum(1 for c in normalized_url if not c.isalnum())

    try:
        parsed = urlparse(normalized_url)
        features["has_https"] = 1 if parsed.scheme == "https" else 0
        features["has_http"] = 1 if parsed.scheme == "http" else 0

        # Extract domain (hostname without port)
        hostname = parsed.hostname or ""
        features["domain"] = hostname
        features["domain_length"] = len(hostname)
        features["path_length"] = len(parsed.path)
        features["query_length"] = len(parsed.query)
        features["fragment_length"] = len(parsed.fragment)

        # Port detection
        features["has_port"] = 1 if parsed.port else 0

        # Subdomain count
        # e.g., 'a.b.example.com' -> parts = ['a', 'b', 'example', 'com'] -> subdomains = 2
        # 'example.com' -> subdomains = 0
        if hostname:
            parts = hostname.split(".")
            if len(parts) > 2:
                features["subdomain_count"] = len(parts) - 2
            else:
                features["subdomain_count"] = 0

            # IP address detection
            try:
                ipaddress.ip_address(hostname)
                features["has_ip_address"] = 1
            except ValueError:
                features["has_ip_address"] = 0

            # Shortener detection
            features["has_shortener_pattern"] = 1 if hostname in KNOWN_SHORTENERS else 0

        features["has_at_symbol"] = 1 if "@" in normalized_url else 0
        features["has_encoded_characters"] = 1 if "%" in normalized_url else 0

    except Exception:
        pass

    return features

def process_single_image(args):
    """
    Worker task to process a single image.
    """
    qr_id, image_filename, image_path, source_folder, label = args

    decoded_text, decode_err = decode_image_qr(image_path)

    if decoded_text is not None:
        decode_status = "success"
        norm_url, extracted_url, content_type = extract_and_normalize_url(decoded_text)
        features = extract_url_features(norm_url)
        decode_error = ""
    else:
        decode_status = "failed"
        norm_url = ""
        extracted_url = ""
        content_type = "unknown"
        features = extract_url_features("")
        decode_error = decode_err or "decode_failed"

    row = {
        "qr_id": qr_id,
        "image_filename": image_filename,
        "image_path": image_path,
        "source_folder": source_folder,
        "label": label,
        "decoded_content": decoded_text if decoded_text is not None else "",
        "normalized_url": norm_url if norm_url else "",
        "content_type": content_type,
        "decode_status": decode_status,
        "url_length": features["url_length"],
        "domain": features["domain"],
        "domain_length": features["domain_length"],
        "path_length": features["path_length"],
        "query_length": features["query_length"],
        "fragment_length": features["fragment_length"],
        "subdomain_count": features["subdomain_count"],
        "has_https": features["has_https"],
        "has_http": features["has_http"],
        "has_ip_address": features["has_ip_address"],
        "has_port": features["has_port"],
        "has_at_symbol": features["has_at_symbol"],
        "has_encoded_characters": features["has_encoded_characters"],
        "has_shortener_pattern": features["has_shortener_pattern"],
        "digit_count": features["digit_count"],
        "special_character_count": features["special_character_count"],
        "decode_error": decode_error,
    }

    return row

def run_pipeline(benign_dir="benign", malicious_dir="malicious", output_dir="qr_decoding_output", limit=None):
    os.makedirs(output_dir, exist_ok=True)

    print("========================================")
    print("STARTING QR CODE -> URL EXTRACTION PIPELINE")
    print("========================================")

    supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    tasks = []
    qr_counter = 1

    # Discover images
    print("Scanning folders for images...")
    
    # Benign (label = 0)
    b_count = 0
    if os.path.exists(benign_dir):
        for entry in os.scandir(benign_dir):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in supported_exts:
                tasks.append((f"QR_{qr_counter:06d}", entry.name, os.path.abspath(entry.path), "benign", 0))
                qr_counter += 1
                b_count += 1
                if limit and b_count >= limit // 2:
                    break

    # Malicious (label = 1)
    m_count = 0
    if os.path.exists(malicious_dir):
        for entry in os.scandir(malicious_dir):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in supported_exts:
                tasks.append((f"QR_{qr_counter:06d}", entry.name, os.path.abspath(entry.path), "malicious", 1))
                qr_counter += 1
                m_count += 1
                if limit and m_count >= limit // 2:
                    break

    total_images = len(tasks)
    print(f"Total images discovered: {total_images}")
    print(f"  Benign images (label 0):    {b_count}")
    print(f"  Malicious images (label 1): {m_count}")

    if total_images == 0:
        print("Error: No images found to process!")
        return

    # Prepare output files
    raw_csv_path = os.path.join(output_dir, "qr_decoded_raw.csv")
    urls_csv_path = os.path.join(output_dir, "qr_decoded_urls.csv")
    failed_csv_path = os.path.join(output_dir, "failed_decodes.csv")
    report_txt_path = os.path.join(output_dir, "qr_decoding_report.txt")
    dup_csv_path = os.path.join(output_dir, "qr_duplicate_report.csv")

    raw_fieldnames = [
        "qr_id", "image_filename", "image_path", "source_folder", "label",
        "decoded_content", "normalized_url", "content_type", "decode_status",
        "url_length", "domain", "domain_length", "path_length", "query_length",
        "fragment_length", "subdomain_count", "has_https", "has_http",
        "has_ip_address", "has_port", "has_at_symbol", "has_encoded_characters",
        "has_shortener_pattern", "digit_count", "special_character_count",
        "decode_error"
    ]

    urls_fieldnames = [
        "qr_id", "image_filename", "source_folder", "label", "decoded_content",
        "normalized_url", "domain", "domain_length", "path_length", "query_length",
        "fragment_length", "subdomain_count", "has_https", "has_http",
        "has_ip_address", "has_port", "has_at_symbol", "has_encoded_characters",
        "has_shortener_pattern", "digit_count", "special_character_count"
    ]

    failed_fieldnames = [
        "qr_id", "image_filename", "source_folder", "label", "image_path", "decode_error"
    ]

    # Initialize CSV files
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f_raw, \
         open(urls_csv_path, "w", newline="", encoding="utf-8") as f_urls, \
         open(failed_csv_path, "w", newline="", encoding="utf-8") as f_failed:
        
        writer_raw = csv.DictWriter(f_raw, fieldnames=raw_fieldnames)
        writer_urls = csv.DictWriter(f_urls, fieldnames=urls_fieldnames)
        writer_failed = csv.DictWriter(f_failed, fieldnames=failed_fieldnames)

        writer_raw.writeheader()
        writer_urls.writeheader()
        writer_failed.writeheader()

    # Statistics accumulators
    successful_decodes = 0
    failed_decodes = 0
    successful_malicious = 0
    successful_benign = 0
    decoded_urls_count = 0
    decoded_non_urls_count = 0
    
    url_occurrences = Counter()
    domain_set = set()
    url_to_labels = defaultdict(lambda: {"benign": 0, "malicious": 0})
    url_to_images = defaultdict(list)

    http_count = 0
    https_count = 0
    ip_count = 0
    shortener_count = 0
    port_count = 0
    at_count = 0
    encoded_count = 0

    workers = max(1, min(cpu_count(), 4))
    print(f"Starting decoding pool with {workers} worker processes...")

    start_time = time.time()
    processed_count = 0
    chunksize = 250

    # Stream write in batches
    batch_raw = []
    batch_urls = []
    batch_failed = []

    with Pool(processes=workers) as pool:
        for row in pool.imap(process_single_image, tasks, chunksize=chunksize):
            processed_count += 1
            batch_raw.append(row)

            if row["decode_status"] == "success":
                successful_decodes += 1
                if row["label"] == 1:
                    successful_malicious += 1
                else:
                    successful_benign += 1

                if row["content_type"] == "url":
                    decoded_urls_count += 1
                    u = row["normalized_url"]
                    url_occurrences[u] += 1
                    if row["label"] == 1:
                        url_to_labels[u]["malicious"] += 1
                    else:
                        url_to_labels[u]["benign"] += 1
                    if len(url_to_images[u]) < 5:  # Store up to 5 sample image names
                        url_to_images[u].append(row["image_filename"])

                    if row["domain"]:
                        domain_set.add(row["domain"])
                    if row["has_http"]:
                        http_count += 1
                    if row["has_https"]:
                        https_count += 1
                    if row["has_ip_address"]:
                        ip_count += 1
                    if row["has_shortener_pattern"]:
                        shortener_count += 1
                    if row["has_port"]:
                        port_count += 1
                    if row["has_at_symbol"]:
                        at_count += 1
                    if row["has_encoded_characters"]:
                        encoded_count += 1

                    url_row = {k: row[k] for k in urls_fieldnames}
                    batch_urls.append(url_row)
                else:
                    decoded_non_urls_count += 1
            else:
                failed_decodes += 1
                failed_row = {k: row[k] for k in failed_fieldnames}
                batch_failed.append(failed_row)

            # Flush batches periodically (every 2,000 images)
            if len(batch_raw) >= 2000:
                with open(raw_csv_path, "a", newline="", encoding="utf-8") as f_raw, \
                     open(urls_csv_path, "a", newline="", encoding="utf-8") as f_urls, \
                     open(failed_csv_path, "a", newline="", encoding="utf-8") as f_failed:
                    
                    csv.DictWriter(f_raw, fieldnames=raw_fieldnames).writerows(batch_raw)
                    if batch_urls:
                        csv.DictWriter(f_urls, fieldnames=urls_fieldnames).writerows(batch_urls)
                    if batch_failed:
                        csv.DictWriter(f_failed, fieldnames=failed_fieldnames).writerows(batch_failed)

                batch_raw.clear()
                batch_urls.clear()
                batch_failed.clear()

            if processed_count % 5000 == 0 or processed_count == total_images:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"[{processed_count:,}/{total_images:,}] ({processed_count/total_images*100:.1f}%) - Speed: {rate:.1f} img/s - Elapsed: {elapsed/60:.1f} min")
                sys.stdout.flush()

    # Flush any remaining items
    if batch_raw:
        with open(raw_csv_path, "a", newline="", encoding="utf-8") as f_raw, \
             open(urls_csv_path, "a", newline="", encoding="utf-8") as f_urls, \
             open(failed_csv_path, "a", newline="", encoding="utf-8") as f_failed:
            
            csv.DictWriter(f_raw, fieldnames=raw_fieldnames).writerows(batch_raw)
            if batch_urls:
                csv.DictWriter(f_urls, fieldnames=urls_fieldnames).writerows(batch_urls)
            if batch_failed:
                csv.DictWriter(f_failed, fieldnames=failed_fieldnames).writerows(batch_failed)

    total_time = time.time() - start_time
    print(f"Decoding complete in {total_time/60:.2f} minutes.")

    # Duplicate analysis
    unique_urls_count = len(url_occurrences)
    duplicate_urls_count = sum(1 for u, count in url_occurrences.items() if count > 1)
    
    print("Writing duplicate report...")
    with open(dup_csv_path, "w", newline="", encoding="utf-8") as f_dup:
        dup_writer = csv.writer(f_dup)
        dup_writer.writerow([
            "normalized_url", "occurrence_count", "benign_count", "malicious_count",
            "is_cross_label_conflict", "sample_image_filenames"
        ])
        for url, count in url_occurrences.most_common():
            if count > 1:
                b_c = url_to_labels[url]["benign"]
                m_c = url_to_labels[url]["malicious"]
                conflict = 1 if (b_c > 0 and m_c > 0) else 0
                sample_imgs = ";".join(url_to_images[url])
                dup_writer.writerow([url, count, b_c, m_c, conflict, sample_imgs])

    # Decode rates
    overall_decode_rate = (successful_decodes / total_images * 100) if total_images > 0 else 0
    malicious_decode_rate = (successful_malicious / m_count * 100) if m_count > 0 else 0
    benign_decode_rate = (successful_benign / b_count * 100) if b_count > 0 else 0

    # Build report text
    report_content = f"""========================================
QR DECODING REPORT
========================================

Dataset statistics:
-------------------
Total images:             {total_images:,}
Malicious images:         {m_count:,}
Benign images:            {b_count:,}

Successfully decoded:     {successful_decodes:,}
Failed to decode:         {failed_decodes:,}

Successful malicious:     {successful_malicious:,}
Successful benign:        {successful_benign:,}

Decoded URLs:             {decoded_urls_count:,}
Decoded non-URLs:         {decoded_non_urls_count:,}

Duplicate URLs:           {duplicate_urls_count:,}
Unique URLs:              {unique_urls_count:,}

Decode rate:
------------
Overall decode rate:      {overall_decode_rate:.2f}%
Malicious decode rate:    {malicious_decode_rate:.2f}%
Benign decode rate:       {benign_decode_rate:.2f}%

URL statistics:
---------------
Unique URLs:              {unique_urls_count:,}
Unique domains:           {len(domain_set):,}
HTTP count:               {http_count:,}
HTTPS count:              {https_count:,}
IP-based URLs:            {ip_count:,}
URL shorteners:           {shortener_count:,}
URLs with ports:          {port_count:,}
URLs containing @:        {at_count:,}
URLs containing encoded characters: {encoded_count:,}

Output files:
-------------
- {raw_csv_path}
- {urls_csv_path}
- {failed_csv_path}
- {dup_csv_path}
- {report_txt_path}
========================================
"""

    with open(report_txt_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_content)

    print("\n" + report_content)

if __name__ == "__main__":
    benign_dir = sys.argv[1] if len(sys.argv) > 1 else "benign"
    malicious_dir = sys.argv[2] if len(sys.argv) > 2 else "malicious"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "qr_decoding_output"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    run_pipeline(benign_dir, malicious_dir, out_dir, limit)
