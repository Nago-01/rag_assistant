import os
import yaml
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
from typing import Union, Optional

from .paths import PUBLICATION_FPATH, ENV_FPATH


def load_publication(file_path = Path) -> str:
    """Loads the publication markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Publication file not found: {file_path}")
    
    # Determine file type by extension
    suffix = file_path.suffix.lower()

    try:
        if suffix in [".md", ".txt"]:
            # try UTF-8 first, then fallback to latin-1
            try:
                return file_path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                return file_path.read_text(encoding="latin-1").strip()
        
        elif suffix == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        
        elif suffix == ".docx":
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    except Exception as e:
        raise IOError(f"Error loading publication file: {e}") from e



def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env() -> None:
    """Loads environment variables from a .env file and checks for required keys.

    Raises:
        AssertionError: If required keys are missing.
    """
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv("GROQ_API_KEY")

    assert api_key, "'api_key' has not been loaded or is not set in the .env file."


def save_text_to_file(
    text: str, filepath: Union[str, Path], header: Optional[str] = None
) -> None:
    """Saves text content to a file, optionally with a header.

    Args:
        text: The content to write.
        filepath: Destination path for the file.
        header: Optional header text to include at the top.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)

    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e
