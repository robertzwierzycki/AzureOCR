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


# Open the JSON file and load data
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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input filename")
    args = parser.parse_args()

    input_file: str = args.input_file

    # Run OCR
    print(datetime.now(), f"Running OCR...")
    ocr_results = await run_ocr(input_file)
    print(ocr_results.content)


if __name__ == "__main__":
    asyncio.run(main())
