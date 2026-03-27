import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MaintenanceChunk:
    """A single chunk of maintenance record text with metadata."""
    text: str
    source_file: str
    aircraft_registration: str
    aircraft_serial: str
    work_order: str
    date: str
    tach_time: str
    chunk_type: str        # "work_item", "header", "parts", "labor"
    item_number: str = ""  # only for work_item chunks


def extract_metadata(text: str) -> dict:
    """Pull key fields from the document header."""
    metadata = {
        "aircraft_registration": "",
        "aircraft_serial": "",
        "work_order": "",
        "date": "",
        "tach_time": "",
    }

    patterns = {
        "aircraft_registration": r"Registration\s+N?o?\.?\s*(N\d+\w*)",
        "aircraft_serial":       r"Serial\s+N?o?\.?\s*(\d{7,})",
        "work_order":            r"WORK\s+ORDER\s+N?O?\.?\s*(\d+)",
        "date":                  r"DATE:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        "tach_time":             r"TT/Hobb/Tach\s+(\d{4,5}\.?\d*)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata[field] = match.group(1).strip()

    return metadata


def chunk_by_work_items(text: str, source_file: str) -> list[MaintenanceChunk]:
    """
    Split maintenance record into chunks by numbered work item.
    Each numbered item (1., 2., 3. etc) becomes its own chunk.
    Header and parts/labor sections become their own chunks too.
    """
    metadata = extract_metadata(text)
    chunks = []

    # Split on numbered work items: "1.", "2.", ... "23."
    # Pattern: newline + number + period at start of line
    item_pattern = re.compile(r'(?=^\d{1,2}\.\s)', re.MULTILINE)
    parts = item_pattern.split(text)

    header_text = parts[0].strip()
    work_item_texts = parts[1:]

    # Header chunk — aircraft identity, dates, admin info
    if header_text:
        chunks.append(MaintenanceChunk(
            text=header_text,
            source_file=source_file,
            chunk_type="header",
            **metadata
        ))

    # One chunk per numbered work item
    for item_text in work_item_texts:
        item_text = item_text.strip()
        if not item_text:
            continue

        # Extract item number from start of text
        num_match = re.match(r'^(\d{1,2})\.', item_text)
        item_number = num_match.group(1) if num_match else ""

        chunks.append(MaintenanceChunk(
            text=item_text,
            source_file=source_file,
            chunk_type="work_item",
            item_number=item_number,
            **metadata
        ))

    return chunks


def load_and_chunk(cleaned_text_path: Path) -> list[MaintenanceChunk]:
    """Load a cleaned text file and return its chunks."""
    text = cleaned_text_path.read_text(encoding="utf-8")
    return chunk_by_work_items(text, source_file=cleaned_text_path.name)


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    test_file = Path("output/cleaned_text")
    files = list(test_file.glob("*.txt"))

    if files:
        chunks = load_and_chunk(files[0])
        print(f"Generated {len(chunks)} chunks from {files[0].name}\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ({chunk.chunk_type}) ---")
            print(f"Aircraft: {chunk.aircraft_registration} | "
                  f"WO: {chunk.work_order} | Date: {chunk.date}")
            print(chunk.text[:200])
            print()