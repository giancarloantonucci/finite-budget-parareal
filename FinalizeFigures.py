"""FinalizeFigures.py

SIAM asks for lines of at least 1 pt in print. Every figure goes in at the SIAM text
width, 5.125 in, so each stroke thinner than 1 pt at that size is widened to 1 pt.
Nothing else in the PDF changes: coordinates, text, colours and data stay as drawn.
Run it after the Julia figure scripts and before compiling the paper; a second run
changes nothing.

Needs numpy, pypdf and PyMuPDF.
Run:     python3 FinalizeFigures.py
Writes:  every PDF in Figures/, in place, and prints each one's thinnest stroke.
"""
from pathlib import Path

import numpy as np
import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject

TEXT_WIDTH = 5.125 * 72  # pt
STROKES = {b'S', b's', b'B', b'B*', b'b', b'b*'}  # operators that stroke a path


def widen(stream, writer, minimum):
    """Draw every stroke whose page width is below `minimum` (page units) at `minimum`."""
    content = ContentStream(stream, writer)
    operations, matrix, width, saved = [], np.eye(3), 1.0, []
    for args, op in content.operations:
        if op == b'q':
            saved.append((matrix.copy(), width))
        elif op == b'Q':
            matrix, width = saved.pop()
        elif op == b'cm':
            a, b, c, d, e, f = map(float, args)
            matrix = matrix @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif op == b'w':
            width = float(args[0])
        elif op in STROKES:
            stretch = np.linalg.svd(matrix[:2, :2], compute_uv=False)[-1]  # the least, to be safe
            needed = minimum / max(stretch, 1e-12)
            if needed > width + 1e-9:
                operations += [([FloatObject(needed)], b'w'), (args, op), ([FloatObject(width)], b'w')]
                continue
        operations.append((args, op))
    content.operations = operations
    return content


for path in sorted((Path(__file__).resolve().parent / 'Figures').glob('*.pdf')):
    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(path))
    for page in writer.pages:
        forms = page.get('/Resources', {}).get('/XObject', {}).values()
        if any(form.get_object().get('/Subtype') == '/Form' for form in forms):
            raise RuntimeError(f'{path.name}: strokes inside form XObjects are not handled')
        minimum = 1.02 * float(page.mediabox.width) / TEXT_WIDTH  # 1 pt at text width, 2% to spare
        page.replace_contents(widen(page.get_contents(), writer, minimum))
    temporary = path.with_suffix('.tmp.pdf')
    writer.write(temporary)
    temporary.replace(path)

    with pymupdf.open(path) as document:
        widths = [drawing['width'] * TEXT_WIDTH / page.rect.width for page in document
                  for drawing in page.get_drawings()
                  if drawing.get('color') is not None and drawing.get('width') is not None]
    if widths:
        assert min(widths) >= 1 - 1e-4, (path.name, min(widths))
        print(f'{path.name}: thinnest stroke {min(widths):.2f} pt at text width ({len(widths)} strokes)')
    else:
        print(f'{path.name}: no strokes')
