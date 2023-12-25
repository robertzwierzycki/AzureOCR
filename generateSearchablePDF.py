import sys
import io
import math
import json
import argparse
import asyncio
from datetime import datetime
from typing import List, Union, Any
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib import pagesizes
from PIL import Image, ImageSequence
from pypdf import PdfWriter, PdfReader
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.ai.formrecognizer import AnalyzeResult, DocumentWord


# Open the JSON file and load config data
with open("config.json", "r") as file:
    config = json.load(file)

endpoint: str = config["DOCUMENT_INTELLIGENCE_URL"]
key: str = config["DOCUMENT_INTELLIGENCE_KEY"]


def dist(p1: DocumentWord, p2: DocumentWord) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


async def load_input_file(input_file: str) -> Union[List[Image.Image], ImageSequence.Iterator]:
    if input_file.lower().endswith(".pdf"):
        return convert_from_path(input_file)
    elif input_file.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")):
        return ImageSequence.Iterator(Image.open(input_file))
    else:
        sys.exit(f"Error: Unsupported file extension {input_file}")


async def run_ocr(input_file: str) -> AnalyzeResult:
    async with DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as client:
        with open(input_file, "rb") as f:
            poller = await client.begin_analyze_document("prebuilt-read", document=f)
        return await poller.result()


def create_searchable_pdf(
    ocr_results: AnalyzeResult,
    image_pages: Union[List[Image.Image], ImageSequence.Iterator],
    output_file: str,
) -> None:
    
    output = PdfWriter()
    default_font: str = "Times-Roman"
    for page_id, page in enumerate(ocr_results.pages):
        ocr_overlay = io.BytesIO()

        # Calculate overlay PDF page size
        if image_pages[page_id].height > image_pages[page_id].width:
            page_scale = float(image_pages[page_id].height) / pagesizes.letter[1]
        else:
            page_scale = float(image_pages[page_id].width) / pagesizes.letter[1]

        page_width = float(image_pages[page_id].width) / page_scale
        page_height = float(image_pages[page_id].height) / page_scale

        scale = (page_width / page.width + page_height / page.height) / 2.0
        pdf_canvas = canvas.Canvas(ocr_overlay, pagesize=(page_width, page_height))

        # Add image into PDF page
        pdf_canvas.drawInlineImage(
            image=image_pages[page_id],
            x=0,
            y=0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True,
        )

        text = pdf_canvas.beginText()
        # Set text rendering mode to invisible
        text.setTextRenderMode(3)
        for word in page.words:
            # Calculate optimal font size
            desired_text_width = (
                max(
                    dist(word.polygon[0], word.polygon[1]),
                    dist(word.polygon[3], word.polygon[2]),
                ) * scale
            )
            desired_text_height = (
                max(
                    dist(word.polygon[1], word.polygon[2]),
                    dist(word.polygon[0], word.polygon[3]),
                ) * scale
            )
            font_size = desired_text_height
            actual_text_width = pdf_canvas.stringWidth(
                word.content, default_font, font_size
            )

            # Calculate text rotation angle
            text_angle = math.atan2(
                (
                    word.polygon[1].y
                    - word.polygon[0].y
                    + word.polygon[2].y
                    - word.polygon[3].y
                ) / 2.0,
                (
                    word.polygon[1].x
                    - word.polygon[0].x
                    + word.polygon[2].x
                    - word.polygon[3].x
                ) / 2.0,
            )
            text.setFont(default_font, font_size)
            text.setTextTransform(
                math.cos(text_angle),
                -math.sin(text_angle),
                math.sin(text_angle),
                math.cos(text_angle),
                word.polygon[3].x * scale,
                page_height - word.polygon[3].y * scale,
            )
            text.setHorizScale(desired_text_width / actual_text_width * 100)
            text.textOut(word.content + " ")

        pdf_canvas.drawText(text)
        pdf_canvas.save()

        # Move to the beginning of the buffer
        ocr_overlay.seek(0)

        # Create a new PDF page
        new_pdf_page = PdfReader(ocr_overlay)
        output.add_page(new_pdf_page.pages[0])

        with open(output_file, "wb") as outputStream:
            output.write(outputStream)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input filename")
    args = parser.parse_args()

    input_file: str = args.input_file
    output_filename: str = ''.join(input_file.split(".")[:-1]) + "_searchable.pdf"

    # Load input file
    print(datetime.now(), f"Loading input file...")
    image_pages = await load_input_file(input_file)

    # Run OCR
    print(datetime.now(), f"Running OCR...")
    ocr_results = await run_ocr(input_file)

    # Create searchable PDF
    print(datetime.now(), f"Creating searchable PDF...")
    create_searchable_pdf(ocr_results, image_pages, output_filename)

    print(datetime.now(), f"Searchable PDF created at {output_filename}")


if __name__ == "__main__":
    asyncio.run(main())
