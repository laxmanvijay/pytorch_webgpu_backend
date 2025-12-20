# PyTorch Inc Backend

Welcome to the PyTorch Inc Backend repository. This project provides custom in-network computing backend support for PyTorch.

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

This package depends on `pytorch_inc_compute` for the `incc` library. There atleast for now, clone both the repos in your local in order to run this. (Instructions on the other package)
This configuration is done using vcpkg's overlay ports in the vcpkg-configuration.json file.

```bash
vcpkg install
pip install -r requirements.txt
```

Everytime `pytorch_inc_compute` changes, it must be recompiled and linked into this repository.

```bash
cd .. && vcpkg install inccompute --overlay-ports=inc_compute && cd ./pytorch_inc_extension && vcpkg install 
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