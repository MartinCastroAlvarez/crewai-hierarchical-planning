# Bundled market dataset

Fictional data for the demo "product launch crew". Every company, product and number
here is made up; any resemblance to a real one is a coincidence. It ships under the
project's MIT license.

| File | Rows | What |
| --- | ---: | --- |
| `products.csv` | 3 | our products: SKU, category, unit cost |
| `segments.csv` | 8 | customer segments per category: size, adoption, willingness to pay |
| `competitors.csv` | 9 | competing products per category with list prices |

The `market_data` tool can only read these three files (or files with the same names
in `LAUNCH_CREW_DATA_DIR`). See `SECURITY.md` for the sandbox rules.
