"""
Download
---------
"""
import os
import requests
from typing import Optional

from pathlib import Path

class FileDownloader:
    def __init__(self, dataset_path = "datasets"):
        self.dataset_path = dataset_path
        
    def save_file_local(self, http_path: str) -> Optional[str]:
        if http_path.endswith(("txt", "md", "rst")):
            filename = http_path.split("/")[-1]
            directory = os.path.join(os.getcwd(), self.dataset_path)
            filepath = os.path.join(directory, filename)
            dir_p = Path(directory)
            file_p = Path(filepath)
            if not file_p.is_file():
                dir_p.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w") as file:
                    response = requests.get(http_path)
                    file.writelines(response.text)
            return filepath
        else:
            raise ValueError("File type not supported. Only .txt, .md, and .rst files are allowed.")
