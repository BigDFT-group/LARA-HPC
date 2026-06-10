#!/usr/bin/env python3
"""Convert HTML documentation to markdown for laraq RAG system.

Usage:
    python utils/html_to_markdown.py <source_dir> <output_dir>

Example:
    python utils/html_to_markdown.py bigdft-docs docs
"""

import html2text
import sys
from pathlib import Path


def convert_html_to_markdown(html_file: Path) -> str:
    """Convert HTML file to markdown, cleaning website chrome."""
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Configure html2text
    converter = html2text.HTML2Text()
    converter.ignore_links = False  # Keep documentation links
    converter.ignore_images = False  # Keep image references
    converter.ignore_emphasis = False  # Keep bold/italic
    converter.body_width = 0  # Don't wrap lines
    converter.skip_internal_links = True  # Skip navigation anchors

    # Convert
    markdown = converter.handle(html_content)

    # Clean up
    return clean_markdown(markdown)


def clean_markdown(text: str) -> str:
    """Remove website navigation and excessive whitespace."""
    import re

    # Remove 3+ consecutive newlines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove common navigation patterns (customize for bigdft-docs if needed)

    return text.strip()


def process_docs(source_dir: Path, output_dir: Path):
    """Process all HTML files from source to output directory."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    # Clear output directory
    if output_dir.exists():
        import shutil
        print(f"Removing existing docs in {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Find all HTML files
    html_files = list(source_dir.rglob("*.html"))
    print(f"Found {len(html_files)} HTML files to convert")

    converted = 0
    for html_file in html_files:
        try:
            # Get relative path
            rel_path = html_file.relative_to(source_dir)

            # Create output path (change .html → .md)
            output_path = output_dir / rel_path.with_suffix('.md')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert
            markdown = convert_html_to_markdown(html_file)

            # Write
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            converted += 1
            print(f"✓ {rel_path}")

        except Exception as e:
            print(f"✗ {rel_path}: {e}")

    print(f"\nConverted {converted}/{len(html_files)} files successfully")
    print(f"Output directory: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2]

    sys.exit(process_docs(source, output))
