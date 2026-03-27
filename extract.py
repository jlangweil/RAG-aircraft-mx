import boto3
import json
import os
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv

load_dotenv()

# Config
RAW_PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("output/extracted_text")
REGION = os.getenv("AWS_REGION", "us-east-1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

textract = boto3.client("textract", region_name=REGION)


def split_pdf_to_pages(pdf_path: Path) -> list[Path]:
    """Split a multi-page PDF into individual page PDFs."""
    reader = PdfReader(pdf_path)
    page_paths = []

    if len(reader.pages) == 1:
        return [pdf_path]  # no splitting needed

    split_dir = Path("data/split_pages") / pdf_path.stem
    split_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        page_path = split_dir / f"page_{i+1:03d}.pdf"
        with open(page_path, "wb") as f:
            writer.write(f)
        page_paths.append(page_path)

    print(f"  Split into {len(page_paths)} pages")
    return page_paths


def extract_text_from_page(pdf_path: Path) -> dict:
    """Send a single page PDF to Textract, return blocks and plain text."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    response = textract.detect_document_text(
        Document={"Bytes": pdf_bytes}
    )

    # Pull out LINE blocks (cleaner than individual WORD blocks)
    lines = [
        block["Text"]
        for block in response["Blocks"]
        if block["BlockType"] == "LINE"
    ]

    # Also capture confidence scores so we know where OCR struggled
    confidence_scores = [
        {"text": block["Text"], "confidence": round(block["Confidence"], 1)}
        for block in response["Blocks"]
        if block["BlockType"] == "LINE"
    ]

    return {
        "plain_text": "\n".join(lines),
        "confidence_scores": confidence_scores,
        "page_count": 1,
    }


def process_pdf(pdf_path: Path):
    """Full pipeline for one PDF: split → extract → save."""
    print(f"\nProcessing: {pdf_path.name}")

    page_paths = split_pdf_to_pages(pdf_path)
    all_text = []
    all_scores = []

    for page_path in page_paths:
        print(f"  Extracting: {page_path.name}")
        result = extract_text_from_page(page_path)
        all_text.append(result["plain_text"])
        all_scores.extend(result["confidence_scores"])

    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

    # Save plain text
    text_output = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    text_output.write_text(combined_text, encoding="utf-8")

    # Save confidence scores alongside for later inspection
    scores_output = OUTPUT_DIR / f"{pdf_path.stem}_confidence.json"
    scores_output.write_text(
        json.dumps(all_scores, indent=2), encoding="utf-8"
    )

    # Quick quality summary
    if all_scores:
        avg_conf = sum(s["confidence"] for s in all_scores) / len(all_scores)
        low_conf = [s for s in all_scores if s["confidence"] < 60]
        print(f"  ✓ Saved to {text_output}")
        print(f"  Avg confidence: {avg_conf:.1f}%")
        if low_conf:
            print(f"  ⚠ {len(low_conf)} low-confidence lines (< 60%)")


def main():
    pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_PDF_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) to process")
    for pdf_path in pdfs:
        process_pdf(pdf_path)

    print("\nDone. Check output/extracted_text/")


if __name__ == "__main__":
    main()