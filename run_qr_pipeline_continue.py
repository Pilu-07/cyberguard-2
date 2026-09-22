import os
import sys
import re
import csv
import time
import ipaddress
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict

import cv2
import zxingcpp
import pyzbar.pyzbar as pyzbar

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "trib.al", "bl.ink", "cur.lv", "bc.vc",
    "po.st", "ity.im", "q-r.to", "u.to", "v.gd", "x.co", "soo.gd"
}

SERIES_URL_REGEX = re.compile(r"^\s*\d+\s+(https?://[^\s\r\n]+)", re.IGNORECASE | re.MULTILINE)
GENERIC_URL_REGEX = re.compile(r"(https?://[^\s\r\n]+)", re.IGNORECASE)

URL_COLS = [
    "qr_id", "image_filename", "source_folder", "label", "decoded_content",
    "normalized_url", "domain", "domain_length", "path_length", "query_length",
    "fragment_length", "subdomain_count", "has_https", "has_http",
    "has_ip_address", "has_port", "has_at_symbol", "has_encoded_characters",
    "has_shortener_pattern", "digit_count", "special_character_count"
]

RAW_COLS = [
    "qr_id", "image_filename", "image_path", "source_folder", "label",
    "decoded_content", "normalized_url", "content_type", "decode_status",
    "url_length", "domain", "domain_length", "path_length", "query_length",
    "fragment_length", "subdomain_count", "has_https", "has_http",
    "has_ip_address", "has_port", "has_at_symbol", "has_encoded_characters",
    "has_shortener_pattern", "digit_count", "special_character_count",
    "decode_error"
]

FAILED_COLS = [
    "qr_id", "image_filename", "source_folder", "label", "image_path", "decode_error"
]

def clean_decoded_text(raw_text):
    if not raw_text:
        return ""
    text = str(raw_text).strip()
    text = re.sub(r"\nName:.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^\d+\s+", "", text).strip()
    return text

def decode_image_qr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "corrupt_or_unreadable_image"

    # 1. zxing-cpp
    try:
        results = zxingcpp.read_barcodes(img)
        for r in results:
            if r.format == zxingcpp.BarcodeFormat.QRCode and r.text:
                return r.text, None
            elif r.text:
                return r.text, None
    except Exception:
        pass

    # 2. OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        decoded_text, _, _ = detector.detectAndDecode(img)
        if decoded_text and decoded_text.strip():
            return decoded_text.strip(), None
    except Exception:
        pass

    # 3. pyzbar
    try:
        barcodes = pyzbar.decode(img)
        for b in barcodes:
            if b.data:
                return b.data.decode("utf-8", errors="replace"), None
    except Exception:
        pass

    # 4. Fallback thresholding & CLAHE
    try:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = zxingcpp.read_barcodes(thresh)
        if res and res[0].text:
            return res[0].text, None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(img)
        res = zxingcpp.read_barcodes(cl_img)
        if res and res[0].text:
            return res[0].text, None
    except Exception:
        pass

    return None, "no_qr_detected_or_unreadable"

def extract_and_normalize_url(raw_text):
    if not raw_text:
        return None, "", "unknown"

    text = clean_decoded_text(raw_text)

    series_match = SERIES_URL_REGEX.search(raw_text)
    if series_match:
        extracted = series_match.group(1).strip()
    elif text.lower().startswith(("http://", "https://")):
        extracted = text.split()[0]
    else:
        generic_match = GENERIC_URL_REGEX.search(raw_text)
        if generic_match:
            extracted = generic_match.group(1).strip()
        else:
            return None, text, "non_url"

    try:
        parsed = urlparse(extracted)
        if not parsed.scheme or not parsed.netloc:
            return None, text, "non_url"

        norm_scheme = parsed.scheme.lower()
        norm_netloc = parsed.netloc.lower()
        norm_url = f"{norm_scheme}://{norm_netloc}{parsed.path}"
        if parsed.query:
            norm_url += f"?{parsed.query}"
        if parsed.fragment:
            norm_url += f"#{parsed.fragment}"

        return norm_url, text, "url"
    except Exception:
        return None, text, "non_url"

def extract_url_features(normalized_url):
    features = {
        "url_length": 0, "domain": "", "domain_length": 0, "path_length": 0,
        "query_length": 0, "fragment_length": 0, "subdomain_count": 0,
        "has_https": 0, "has_http": 0, "has_ip_address": 0, "has_port": 0,
        "has_at_symbol": 0, "has_encoded_characters": 0, "has_shortener_pattern": 0,
        "digit_count": 0, "special_character_count": 0
    }
    if not normalized_url:
        return features

    features["url_length"] = len(normalized_url)
    features["digit_count"] = sum(1 for c in normalized_url if c.isdigit())
    features["special_character_count"] = sum(1 for c in normalized_url if not c.isalnum())
    features["has_at_symbol"] = 1 if "@" in normalized_url else 0
    features["has_encoded_characters"] = 1 if "%" in normalized_url else 0

    try:
        parsed = urlparse(normalized_url)
        features["has_https"] = 1 if parsed.scheme == "https" else 0
        features["has_http"] = 1 if parsed.scheme == "http" else 0
        hostname = parsed.hostname or ""
        features["domain"] = hostname
        features["domain_length"] = len(hostname)
        features["path_length"] = len(parsed.path)
        features["query_length"] = len(parsed.query)
        features["fragment_length"] = len(parsed.fragment)
        features["has_port"] = 1 if parsed.port else 0

        if hostname:
            parts = hostname.split(".")
            features["subdomain_count"] = max(len(parts) - 2, 0) if len(parts) > 2 else 0
            try:
                ipaddress.ip_address(hostname)
                features["has_ip_address"] = 1
            except ValueError:
                features["has_ip_address"] = 0
            features["has_shortener_pattern"] = 1 if hostname in KNOWN_SHORTENERS else 0
    except Exception:
        pass

    return features

def worker_process_image(args):
    qr_id, image_filename, image_path, source_folder, label = args
    decoded_raw, decode_err = decode_image_qr(image_path)

    if decoded_raw is not None:
        decode_status = "success"
        norm_url, clean_content, content_type = extract_and_normalize_url(decoded_raw)
        features = extract_url_features(norm_url)
        decode_error = ""
    else:
        decode_status = "failed"
        norm_url = ""
        clean_content = ""
        content_type = "unknown"
        features = extract_url_features("")
        decode_error = decode_err or "decode_failed"

    raw_record = {
        "qr_id": qr_id,
        "image_filename": image_filename,
        "image_path": image_path,
        "source_folder": source_folder,
        "label": label,
        "decoded_content": clean_content,
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

    url_record = None
    if decode_status == "success" and content_type == "url":
        url_record = {
            "qr_id": qr_id,
            "image_filename": image_filename,
            "source_folder": source_folder,
            "label": label,
            "decoded_content": clean_content,
            "normalized_url": norm_url,
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
        }

    return raw_record, url_record

def process_folder(folder_path, source_folder, label, output_csv_path, id_prefix, existing_done_files):
    if not os.path.exists(folder_path):
        print(f"\n[{source_folder.upper()}] Folder '{folder_path}' does not exist. Using existing decoded entries ({len(existing_done_files):,}).")
        return

    supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_exts])
    total_in_folder = len(all_files)
    
    pending_files = [f for f in all_files if f not in existing_done_files]
    print(f"\n[{source_folder.upper()}] Total: {total_in_folder:,} | Already processed: {len(existing_done_files):,} | Pending to process: {len(pending_files):,}")

    if not pending_files:
        print(f"[{source_folder.upper()}] Nothing to do, all files already processed.")
        return

    # Check if target CSV exists to append or write header
    write_header = not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0

    start_num = len(existing_done_files) + 1
    tasks = []
    for idx, fname in enumerate(pending_files, start=start_num):
        img_id = f"{id_prefix}_{idx:06d}"
        img_path = os.path.abspath(os.path.join(folder_path, fname))
        tasks.append((img_id, fname, img_path, source_folder, label))

    workers = max(1, min(cpu_count(), 4))
    print(f"[{source_folder.upper()}] Starting pool with {workers} workers for {len(tasks):,} tasks...")

    t0 = time.time()
    batch_urls = []
    raw_records_accumulator = []
    processed_count = 0
    total_pending = len(tasks)

    # Open file in append mode
    with open(output_csv_path, "a", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=URL_COLS)
        if write_header:
            writer.writeheader()

        with Pool(processes=workers) as pool:
            for raw_rec, url_rec in pool.imap(worker_process_image, tasks, chunksize=100):
                processed_count += 1
                if url_rec:
                    batch_urls.append(url_rec)

                # Save raw record to temporary list/file for final merge
                # We can store in memory or append to folder raw log
                raw_records_accumulator.append(raw_rec)

                if len(batch_urls) >= 1000:
                    writer.writerows(batch_urls)
                    f_out.flush()
                    batch_urls.clear()

                if processed_count % 5000 == 0 or processed_count == total_pending:
                    elapsed = time.time() - t0
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    rem = (total_pending - processed_count) / speed if speed > 0 else 0
                    print(f"  [{source_folder.upper()}] {processed_count:,}/{total_pending:,} ({processed_count/total_pending*100:.1f}%) - Speed: {speed:.1f} img/s - ETA: {rem/60:.1f} min")
                    sys.stdout.flush()

        if batch_urls:
            writer.writerows(batch_urls)
            f_out.flush()
            batch_urls.clear()

    total_time = time.time() - t0
    print(f"[{source_folder.upper()}] Completed {processed_count:,} images in {total_time/60:.2f} minutes.")
    return raw_records_accumulator

def generate_final_combined_outputs(output_dir="qr_decoding_output", benign_dir="benign", malicious_dir="malicious"):
    print("\n========================================")
    print("ASSEMBLING FINAL DATASETS & QUALITY REPORT")
    print("========================================")

    benign_csv = os.path.join(output_dir, "benign_decoded.csv")
    malicious_csv = os.path.join(output_dir, "malicious_decoded.csv")
    urls_csv = os.path.join(output_dir, "qr_decoded_urls.csv")
    raw_csv = os.path.join(output_dir, "qr_decoded_raw.csv")
    failed_csv = os.path.join(output_dir, "failed_decodes.csv")
    dup_csv = os.path.join(output_dir, "qr_duplicate_report.csv")
    report_txt = os.path.join(output_dir, "qr_decoding_report.txt")

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

    total_benign_urls = 0
    total_malicious_urls = 0

    # 1. Merge into qr_decoded_urls.csv and gather stats
    print("Writing qr_decoded_urls.csv ...")
    with open(urls_csv, "w", newline="", encoding="utf-8-sig") as f_urls:
        writer_urls = csv.DictWriter(f_urls, fieldnames=URL_COLS)
        writer_urls.writeheader()

        # Read benign
        if os.path.exists(benign_csv):
            with open(benign_csv, "r", newline="", encoding="utf-8-sig") as fb:
                r = csv.DictReader(fb)
                for row in r:
                    total_benign_urls += 1
                    u = row["normalized_url"]
                    if u:
                        url_occurrences[u] += 1
                        url_to_labels[u]["benign"] += 1
                        if len(url_to_images[u]) < 5:
                            url_to_images[u].append(row["image_filename"])
                        if row.get("domain"):
                            domain_set.add(row["domain"])
                        if row.get("has_http") == "1":
                            http_count += 1
                        if row.get("has_https") == "1":
                            https_count += 1
                        if row.get("has_ip_address") == "1":
                            ip_count += 1
                        if row.get("has_shortener_pattern") == "1":
                            shortener_count += 1
                        if row.get("has_port") == "1":
                            port_count += 1
                        if row.get("has_at_symbol") == "1":
                            at_count += 1
                        if row.get("has_encoded_characters") == "1":
                            encoded_count += 1
                    writer_urls.writerow(row)

        # Read malicious
        if os.path.exists(malicious_csv):
            with open(malicious_csv, "r", newline="", encoding="utf-8-sig") as fm:
                r = csv.DictReader(fm)
                for row in r:
                    total_malicious_urls += 1
                    u = row["normalized_url"]
                    if u:
                        url_occurrences[u] += 1
                        url_to_labels[u]["malicious"] += 1
                        if len(url_to_images[u]) < 5:
                            url_to_images[u].append(row["image_filename"])
                        if row.get("domain"):
                            domain_set.add(row["domain"])
                        if row.get("has_http") == "1":
                            http_count += 1
                        if row.get("has_https") == "1":
                            https_count += 1
                        if row.get("has_ip_address") == "1":
                            ip_count += 1
                        if row.get("has_shortener_pattern") == "1":
                            shortener_count += 1
                        if row.get("has_port") == "1":
                            port_count += 1
                        if row.get("has_at_symbol") == "1":
                            at_count += 1
                        if row.get("has_encoded_characters") == "1":
                            encoded_count += 1
                    writer_urls.writerow(row)

    total_urls = total_benign_urls + total_malicious_urls
    print(f"Saved {total_urls:,} total URLs ({total_benign_urls:,} benign, {total_malicious_urls:,} malicious) -> {urls_csv}")

    # 2. Build duplicate report
    unique_urls_count = len(url_occurrences)
    duplicate_urls_count = sum(1 for u, count in url_occurrences.items() if count > 1)
    print(f"Writing qr_duplicate_report.csv ({duplicate_urls_count:,} duplicate URLs)...")
    with open(dup_csv, "w", newline="", encoding="utf-8") as f_dup:
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
                samples = ";".join(url_to_images[url])
                dup_writer.writerow([url, count, b_c, m_c, conflict, samples])

    # 3. Build qr_decoded_raw.csv and failed_decodes.csv
    print("Building clean qr_decoded_raw.csv and failed_decodes.csv...")
    failed_rows = []
    total_images_in_dataset = 0
    total_benign_images = sum(1 for _ in os.scandir(benign_dir)) if os.path.exists(benign_dir) else 0
    total_malicious_images = sum(1 for _ in os.scandir(malicious_dir)) if os.path.exists(malicious_dir) else 0
    total_images_in_dataset = total_benign_images + total_malicious_images

    # Reconstruct raw records directly from benign_decoded.csv and malicious_decoded.csv
    with open(raw_csv, "w", newline="", encoding="utf-8") as f_raw, \
         open(failed_csv, "w", newline="", encoding="utf-8") as f_fail:
        
        writer_raw = csv.DictWriter(f_raw, fieldnames=RAW_COLS)
        writer_fail = csv.DictWriter(f_fail, fieldnames=FAILED_COLS)
        writer_raw.writeheader()
        writer_fail.writeheader()

        def stream_to_raw(csv_file, folder_name, folder_path):
            if not os.path.exists(csv_file):
                return
            with open(csv_file, "r", newline="", encoding="utf-8-sig") as f_in:
                reader = csv.DictReader(f_in)
                for r in reader:
                    img_name = r["image_filename"]
                    full_p = os.path.abspath(os.path.join(folder_path, img_name))
                    norm = r["normalized_url"]
                    content = r["decoded_content"]
                    is_success = bool(norm or content)
                    status = "success" if is_success else "failed"
                    c_type = "url" if norm else ("non_url" if content else "unknown")
                    err = "" if is_success else "decode_failed"

                    raw_item = {
                        "qr_id": r["qr_id"],
                        "image_filename": img_name,
                        "image_path": full_p,
                        "source_folder": r["source_folder"],
                        "label": r["label"],
                        "decoded_content": content,
                        "normalized_url": norm,
                        "content_type": c_type,
                        "decode_status": status,
                        "url_length": len(norm),
                        "domain": r["domain"],
                        "domain_length": r["domain_length"],
                        "path_length": r["path_length"],
                        "query_length": r["query_length"],
                        "fragment_length": r["fragment_length"],
                        "subdomain_count": r["subdomain_count"],
                        "has_https": r["has_https"],
                        "has_http": r["has_http"],
                        "has_ip_address": r["has_ip_address"],
                        "has_port": r["has_port"],
                        "has_at_symbol": r["has_at_symbol"],
                        "has_encoded_characters": r["has_encoded_characters"],
                        "has_shortener_pattern": r["has_shortener_pattern"],
                        "digit_count": r["digit_count"],
                        "special_character_count": r["special_character_count"],
                        "decode_error": err
                    }
                    writer_raw.writerow(raw_item)
                    if not is_success:
                        writer_fail.writerow({
                            "qr_id": r["qr_id"],
                            "image_filename": img_name,
                            "source_folder": r["source_folder"],
                            "label": r["label"],
                            "image_path": full_p,
                            "decode_error": err
                        })

        stream_to_raw(benign_csv, "benign", benign_dir)
        stream_to_raw(malicious_csv, "malicious", malicious_dir)

    # 4. Generate report text
    successful_decodes = total_urls
    failed_decodes = total_images_in_dataset - successful_decodes
    overall_decode_rate = (successful_decodes / total_images_in_dataset * 100) if total_images_in_dataset > 0 else 0
    malicious_decode_rate = (total_malicious_urls / total_malicious_images * 100) if total_malicious_images > 0 else 0
    benign_decode_rate = (total_benign_urls / total_benign_images * 100) if total_benign_images > 0 else 0

    report_content = f"""========================================
QR DECODING COMPLETE
========================================

Dataset statistics:
-------------------
Total images:             {total_images_in_dataset:,}
Malicious images:         {total_malicious_images:,}
Benign images:            {total_benign_images:,}

Successfully decoded:     {successful_decodes:,}
Failed to decode:         {failed_decodes:,}

Successful malicious:     {total_malicious_urls:,}
Successful benign:        {total_benign_urls:,}

Decoded URLs:             {total_urls:,}
Decoded non-URLs:         0

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
- {os.path.abspath(benign_csv)}
- {os.path.abspath(malicious_csv)}
- {os.path.abspath(urls_csv)}
- {os.path.abspath(raw_csv)}
- {os.path.abspath(failed_csv)}
- {os.path.abspath(dup_csv)}
- {os.path.abspath(report_txt)}
========================================
"""
    with open(report_txt, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_content)

    print("\n" + report_content)

def main():
    benign_dir = "benign"
    malicious_dir = "malicious"
    out_dir = "qr_decoding_output"
    os.makedirs(out_dir, exist_ok=True)

    benign_csv = os.path.join(out_dir, "benign_decoded.csv")
    malicious_csv = os.path.join(out_dir, "malicious_decoded.csv")

    # Step 1: Scan existing benign_decoded.csv
    done_benign = set()
    if os.path.exists(benign_csv) and os.path.getsize(benign_csv) > 0:
        with open(benign_csv, "r", newline="", encoding="utf-8-sig") as fb:
            r = csv.DictReader(fb)
            for row in r:
                fn = row.get("image_filename")
                if fn:
                    done_benign.add(fn)
    print(f"Found {len(done_benign):,} previously decoded benign images in {benign_csv}")

    # Process remaining benign images
    process_folder(
        folder_path=benign_dir,
        source_folder="benign",
        label=0,
        output_csv_path=benign_csv,
        id_prefix="QR_BENIGN",
        existing_done_files=done_benign
    )

    # Step 2: Scan existing malicious_decoded.csv
    done_malicious = set()
    if os.path.exists(malicious_csv) and os.path.getsize(malicious_csv) > 0:
        with open(malicious_csv, "r", newline="", encoding="utf-8-sig") as fm:
            r = csv.DictReader(fm)
            for row in r:
                fn = row.get("image_filename")
                if fn:
                    done_malicious.add(fn)
    print(f"Found {len(done_malicious):,} previously decoded malicious images in {malicious_csv}")

    # Process remaining malicious images
    process_folder(
        folder_path=malicious_dir,
        source_folder="malicious",
        label=1,
        output_csv_path=malicious_csv,
        id_prefix="QR_MAL",
        existing_done_files=done_malicious
    )

    # Step 3: Produce unified files and reports
    generate_final_combined_outputs(output_dir=out_dir, benign_dir=benign_dir, malicious_dir=malicious_dir)

if __name__ == "__main__":
    main()
