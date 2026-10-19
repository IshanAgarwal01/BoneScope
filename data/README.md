# Data (not committed)

This folder is gitignored. Download the dataset yourself:

- Shen D. et al., "A large dataset of bone marrow cells in myelodysplastic
  syndrome for classification systems," *Scientific Data* 12, 1849 (2025).
- figshare: https://doi.org/10.6084/m9.figshare.28737170

Unpack it so the class folders (or the archive's own top-level folder) sit under
`data/raw/`, then run:

```bash
python scripts/inspect_data.py --root data/raw
```

The inspector reads the class count and names from the files; nothing is hard-coded.
