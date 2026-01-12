# Sphinx Documentation

This directory contains the Sphinx-based API reference documentation for multicalibration, which is automatically generated from Python docstrings.

## Building Documentation Locally

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
# Also install the package itself
pip install -e ..
```

### Building HTML Documentation

```bash
make html
```

The HTML documentation will be generated in `build/html/`.

### Viewing the Documentation

After building, you can view the documentation by opening `build/html/index.html` in your browser, or by running a local web server:

```bash
cd build/html
python3 -m http.server 8000
```

Then navigate to http://localhost:8000

### Other Build Formats

Sphinx supports various output formats:

```bash
make pdf      # Build PDF documentation
make epub     # Build ePub documentation
make latexpdf # Build LaTeX PDF
make clean    # Clean the build directory
```

## ReadTheDocs Integration

This documentation is automatically built and hosted on ReadTheDocs whenever you push to the repository. The configuration is in `../.readthedocs.yaml`.

Once set up on ReadTheDocs, the documentation will be available at:
https://mcgrad.readthedocs.io/

## Structure

- `source/conf.py` - Sphinx configuration file
- `source/index.rst` - Main documentation entry point
- `source/api/` - API reference RST files that use autodoc to generate documentation from docstrings
- `build/` - Generated documentation (not committed to git)

## Updating API Documentation

The API documentation is automatically generated from the Python source code docstrings.

**Adding a new module:**

1. Create `source/api/<module_name>.rst`:
   ```rst
   Module Name
   ===========

   .. automodule:: multicalibration.<module_name>
      :members:
      :undoc-members:
      :show-inheritance:
   ```

   **That's it!** The `automodule` directive with `:members:` automatically
   documents all classes and functions. No need to list them manually.

2. Add it to `source/index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2

      api/methods
      api/metrics
      api/<module_name>  # Add here
   ```

The current setup uses:
- **Google-style docstrings** (via `napoleon` extension)
- **Automatic type hint extraction** (via `sphinx-autodoc-typehints`)
- **ReadTheDocs theme** for consistent styling
- **automodule directive** - automatically discovers and documents all public classes/functions
