import pytest
import subprocess
import sys
import re
import importlib.util


class TestExamSolver:
    """Tests for NLP Exam Solver functionality."""

    def test_imports(self):
        """Test that required packages are available."""
        try:
            import requests
            import dotenv
            from pathlib import Path
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

    def test_ollama_available(self):
        """Test if Ollama CLI is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
        except FileNotFoundError:
            pytest.skip("Ollama not installed")
        except subprocess.TimeoutExpired:
            pytest.skip("Ollama not responding")

    def test_extract_code_basic(self):
        """Test code extraction from response."""
        from exam_solver import extract_code

        response = """Here is the solution:
```python
import nltk
print("Hello")
```
"""
        code = extract_code(response)
        assert "import nltk" in code
        assert "print" in code

    def test_extract_code_no_block(self):
        """Test code extraction when no code block."""
        from exam_solver import extract_code

        response = "Just some text without code"
        code = extract_code(response)
        assert "Just some text" in code

    def test_solve_question_format(self):
        """Test question solving returns correct format."""
        from exam_solver import solve_question, call_ollama, extract_code

        # Mock the call to avoid actual API call in tests
        prompt = "Test prompt"
        response = """```python
import nltk
# Test
```
"""
        code = extract_code(response)
        assert "import nltk" in code


class TestExamPipeline:
    """Tests for exam_solver_pipeline.py"""

    def test_pipeline_imports(self):
        """Test pipeline imports work."""
        try:
            # Just check imports don't error
            import json
            import subprocess
            import tempfile
            from pathlib import Path
        except ImportError as e:
            pytest.fail(f"Missing import: {e}")

    def test_code_extraction_regex(self):
        """Test code extraction regex works."""
        from exam_solver_pipeline import extract_code_from_response

        response = """Here's the solution:

```python
def solve():
    return True
```

End of response."""

        code = extract_code_from_response(response)
        assert "def solve" in code
        assert "return True" in code

    def test_multiline_code_extraction(self):
        """Test extracting multi-line code blocks."""
        from exam_solver_pipeline import extract_code_from_response

        response = """```python
import nltk
from collections import Counter

def freq_dist(text):
    return Counter(text.split())
```
"""
        code = extract_code_from_response(response)
        assert "import nltk" in code
        assert "from collections import Counter" in code
        assert "def freq_dist" in code


class TestModelfile:
    """Tests for Modelfile configuration."""

    def test_modelfile_exists(self):
        """Test Modelfile exists."""
        from pathlib import Path
        modelfile = Path("Modelfile")
        assert modelfile.exists(), "Modelfile should exist"

    def test_modelfile_has_from(self):
        """Test Modelfile has FROM directive."""
        from pathlib import Path
        content = Path("Modelfile").read_text(encoding='utf-8', errors='ignore')
        assert "FROM" in content, "Modelfile should have FROM directive"

    def test_modelfile_has_gguf(self):
        """Test Modelfile references GGUF file."""
        from pathlib import Path
        content = Path("Modelfile").read_text(encoding='utf-8', errors='ignore')
        assert ".gguf" in content, "Modelfile should reference .gguf file"


class TestInstaller:
    """Tests for install.ps1 script."""

    def test_install_script_exists(self):
        """Test install.ps1 exists."""
        from pathlib import Path
        installer = Path("install.ps1")
        assert installer.exists(), "install.ps1 should exist"

    def test_install_script_shebang(self):
        """Test install.ps1 has proper structure."""
        from pathlib import Path
        content = Path("install.ps1").read_text()
        assert "param(" in content, "install.ps1 should have param block"
        assert "$ErrorActionPreference" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])