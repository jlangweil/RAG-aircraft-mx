import anthropic
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env file into environment

EXTRACTED_DIR = Path("output/extracted_text")
CLEANED_DIR = Path("output/cleaned_text")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLEANUP_SYSTEM_PROMPT = """You are an expert in aviation maintenance records and FAA repair station 
documentation. You will be given raw OCR text extracted from scanned aircraft maintenance work orders.

Your job is to clean up OCR errors while preserving the meaning and technical accuracy of the original.

Rules:
- Fix obvious OCR errors using aviation maintenance context (e.g. "Magneto" not "Miacneto")
- Preserve ALL part numbers, serial numbers, N-numbers, and AD numbers exactly as written
  — do not correct these even if they look wrong, they may be accurate
- Preserve all dates and dollar amounts exactly
- Where a line is too garbled to interpret confidently, output it with a [UNCERTAIN] prefix
- Do not add information that wasn't in the original
- Do not remove any line items, even if garbled
- Output only the cleaned text, no commentary"""


def cleanup_extracted_text(text: str, source_filename: str) -> str:
    """Send extracted text through Claude for OCR cleanup."""
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=CLEANUP_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Please clean up the following OCR-extracted text from aircraft 
maintenance work order: {source_filename}

RAW OCR TEXT:
{text}"""
            }
        ]
    )
    
    return response.content[0].text


def process_all_extracted():
    txt_files = list(EXTRACTED_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"No extracted text files found in {EXTRACTED_DIR}")
        return
    
    print(f"Found {len(txt_files)} file(s) to clean")
    
    for txt_path in txt_files:
        print(f"\nCleaning: {txt_path.name}")
        
        raw_text = txt_path.read_text(encoding="utf-8")
        
        # Load confidence scores if available
        scores_path = EXTRACTED_DIR / txt_path.name.replace(".txt", "_confidence.json")
        if scores_path.exists():
            scores = json.loads(scores_path.read_text())
            low_conf_count = sum(1 for s in scores if s["confidence"] < 60)
            print(f"  {low_conf_count} low-confidence lines to clean")
        
        cleaned = cleanup_extracted_text(raw_text, txt_path.name)
        
        # Save cleaned version
        out_path = CLEANED_DIR / txt_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  ✓ Saved to {out_path}")


if __name__ == "__main__":
    process_all_extracted()