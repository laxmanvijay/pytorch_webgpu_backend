# PyTorch Inc Backend

Welcome to the PyTorch Webgpu Backend repository. This project provides custom webgpu computing backend support for PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install the necessary dependencies, install vcpkg first.

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

./bootstrap-vcpkg.sh  # Linux/macOS
.\bootstrap-vcpkg.bat  # Windows
```

Add the install directory to `VCPKG_ROOT`

Then create a virtual environment (At the time of writing, pytorch doesn't yet support python >3.13. Therefore it is better to use something less that 3.13):

```bash
python3.12 -m venv venv
source ./venv/bin/activate
```

Install the dependencies for both c++ and python:

```bash
vcpkg install
pip install -r requirements.txt
```

## Usage

The setup.py by default points to the default vcpkg install folder: `vcpkg_installed` for include and lib folders.
The triplet is detected from within setup.py.

To build and install the extension:

```bash
python setup.py develop
```

Run the `test.py` file to see a simple demo.

```bash
python test.py
```

### Potential Errors and fixes:

* ImportError: dlopen: symbol not found in flat namespace - Linker error: Check if any new files that are added in c++ are included in the build.