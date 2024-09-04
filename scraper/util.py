import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def save_pdf(paper_metadata: Dict[str, Any], filepath: str) -> None:
    """
    Save a PDF file of a paper.

    Args:
        paper_metadata (Dict[str, Any]): A dictionary with the paper metadata. Must
            contain the `doi` key.
        filepath (str): Path to the file to be saved.
    """
    if not isinstance(paper_metadata, Dict):
        raise TypeError(f"paper_metadata must be a dict, not {type(paper_metadata)}.")
    if "doi" not in paper_metadata.keys():
        raise KeyError("paper_metadata must contain the key 'doi'.")
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}.")
    if not filepath.endswith(".pdf"):
        raise ValueError("Please provide a filepath with .pdf extension.")
    if not Path(filepath).parent.exists():
        raise ValueError(f"The folder: {Path(filepath).parent} seems to not exist.")

    url = f"https://doi.org/{paper_metadata['doi']}"
    try:
        response = requests.get(url, timeout=60)
    except Exception:
        print(f"Could not download {url}.")
        return f"Could not download {url}."

    soup = BeautifulSoup(response.text, features="lxml")

    metas = soup.find("meta", {"name": "citation_pdf_url"})
    if metas is None:
      print(
            f"Could not find PDF for: {url} (either there's a paywall or the host "
            "blocks PDF scraping)."
        )
      return f"Could not find PDF for: {url} (either there's a paywall or the host blocks PDF scraping)."
    pdf_url = metas.attrs.get("content")

    try:
        response = requests.get(pdf_url, timeout=60)
    except Exception:
        print(f"Could not download {pdf_url}.")
        return f"Could not download {pdf_url}."
    
    
    with open(filepath, "wb+") as f:
        f.write(response.content)
    return  filepath

# Define the function to be applied to each row
def save_paper_pdf(row, catalog, schema, volume_folder, topic):

    if row['doi'] is  None:
        return "Could not find DOI"
    elif "\n" in row['doi']:
        # get the first DOI
        row['doi'] = row['doi'].split("\n")[0]

    paper_data = {'doi': row['doi']}
    
    # Create the directory path
    directory_path = f'/Volumes/{catalog}/{schema}/{volume_folder}/{topic}/'
    os.makedirs(directory_path, exist_ok=True)
    
    # Generate a file name using DOI or another identifier to avoid conflicts
    pdf_filename = f"{row['doi'].replace('/', '_')}.pdf"
    pdf_path = os.path.join(directory_path, pdf_filename)
    
    # Save the PDF
    saved_path = save_pdf(paper_data, filepath=pdf_path)
    
    return saved_path