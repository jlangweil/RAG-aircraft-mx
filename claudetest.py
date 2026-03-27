import anthropic
import base64
import sys
import fitz  # pymupdf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}

EXTRACTION_PROMPT = """This is a scanned aircraft maintenance work order from an FAA repair station. 
Please transcribe ALL text exactly as written, preserving:
- All numbered work items and their full descriptions
- Part numbers and serial numbers exactly as written
- Dates, tach times, and dollar amounts
- Mechanic notes and squawk descriptions
- Labor breakdowns and totals
- All header fields (aircraft registration, serial number, work order number)

Output only the transcribed text, no commentary."""


def pdf_page_to_base64(pdf_path: Path, page_number: int = 0) -> str:
    """Convert a single page of a PDF to base64-encoded JPEG using pymupdf."""
    doc = fitz.open(str(pdf_path))

    if page_number >= len(doc):
        raise ValueError(f"Page {page_number} doesn't exist — PDF has {len(doc)} pages")

    page = doc[page_number]

    # 200 DPI equivalent — matrix scale factor of 200/72
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat)

    img_path = Path("data/vision_test_page.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(img_path))

    with open(img_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def extract_with_vision(image_b64: str, model_key: str) -> str:
    """Send image to Claude vision and return extracted text."""
    client = anthropic.Anthropic()
    model = MODELS[model_key]

    print(f"  Extracting with {model_key} ({model})...")

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ],
            }
        ],
    )

    return response.content[0].text


def main():
    # Accept PDF path as argument or fall back to first PDF in raw_pdfs
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdfs = list(Path("data/raw_pdfs").glob("*.pdf"))
        if not pdfs:
            print("No PDFs found in data/raw_pdfs/")
            sys.exit(1)
        pdf_path = pdfs[0]

    print(f"PDF: {pdf_path.name}")
    print(f"Converting page 2 to image at 200 DPI...")

    image_b64 = pdf_page_to_base64(pdf_path, page_number=1)
    print(f"  Image ready\n")

    output_dir = Path("output/vision_extract")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in ["sonnet"]:
        text = extract_with_vision(image_b64, model_key)

        out_path = output_dir / f"{pdf_path.stem}_page1_{model_key}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved to {out_path}\n")

    print("Done. Check output/vision_extract/ for both results.")


if __name__ == "__main__":
    main()