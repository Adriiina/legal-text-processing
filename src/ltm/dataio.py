import gzip
import json
from typing import Iterator, Dict, Any

def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL.GZ file line by line.
    Each line should be a JSON object.
    
    Args:
        path: Path to .jsonl.gz file
    
    Yields:
        dict: Parsed JSON object
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipped invalid line: {e}")
                    continue

