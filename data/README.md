# Data

<!-- Drafted with Claude (Anthropic). Asked for: a place to record where the data came from, its licence and how it was cleaned (Requirements 4.5 and 4.15), then filled in for the CEPII Gravity data when stage 1 was written. -->

| File | Source | Downloaded on | Licence | In the repository? |
|---|---|---|---|---|
| `raw/Gravity_V202211.csv` | CEPII Gravity database, version V202211 ([page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8), [zip](https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip)) | Before 3 October 2026 (Yu's copy). Stage 1 prints the date of each new download. | Etalab Open Licence 2.0 | No: 1.25 GB, over GitHub's 100 MB limit. `code/01_get_data.ipynb` downloads it into `data/raw/`, which `.gitignore` leaves out (Requirement 4.15). |
| `gravity_extract.csv` | Cut from the file above by `code/01_get_data.ipynb` | | Etalab Open Licence 2.0 | Yes, 8.3 MB |

## The CEPII file

- 1,251,418,200 bytes, SHA-256 `c5611acb52288c537786dda86f2c3319cdd04cc94e8c9328dde936510a17c739`. Stage 1 stops if a download differs.
- 4,699,296 rows and 87 columns, years 1948 to 2021. Every year has 63,504 rows: 252 × 252 country identifiers, a country paired with itself included.
- Licence: the CEPII documentation (version of 12 October 2023, pp. 7–8) says the data "is distributed under the Etalab Open Licence 2.0, meaning that it can be freely used, modified, and shared as long as a proper reference is made to the source". This is why the extract can be shared here, with the reference below.
- Reference asked for by CEPII: Conte, M., P. Cotterlaz and T. Mayer (2022), "The CEPII Gravity database", CEPII Working Paper N°2022-05, July 2022.

## The extract: `gravity_extract.csv`

- Every row of 2017 (the development year) and of 2019 (the analysis year), 63,504 each and 127,008 in all, with the 16 columns below.
- The values are those of the CEPII file, copied as text. A missing value is an empty field.
- 8,258,467 bytes, SHA-256 `5122d30133ddb99ccba3bfff0b0b4d67a8713ecdf35b2ee540def1f1666bcf93`. Running stage 1 again on the same CEPII file writes the same file, so `git status` then shows no change.

One row is one ordered pair of countries in one year, from the origin (`_o`) to the destination (`_d`). Contents and units are those of the variable list in the CEPII documentation (Table 1).

| Column | Content |
|---|---|
| `year` | Year |
| `country_id_o`, `country_id_d` | Gravity country identifier: the ISO3 code, numbered where a country's territory changed, for example `DEU.1` (West Germany) and `DEU.2` (unified Germany). With `year`, they identify the row |
| `iso3_o`, `iso3_d` | ISO3 alphabetic code |
| `country_exists_o`, `country_exists_d` | 1 if the country exists in that year |
| `distw_harmonic` | Population-weighted distance between the most populated cities, harmonic mean, in km |
| `contig` | 1 if the countries are contiguous |
| `comlang_off` | 1 if they share a common official or primary language |
| `col_dep_ever` | 1 if the pair was ever in a colonial or dependency relationship (including before 1948) |
| `comcol` | 1 if they share a common colonizer after 1945 |
| `fta_wto` | 1 if the pair is in a regional trade agreement (source: WTO, supplemented by Thierry Mayer) |
| `gdp_o`, `gdp_d` | GDP, in thousands of current US dollars |
| `tradeflow_baci` | Trade flow from origin to destination, in thousands of current US dollars (source: BACI). Empty when no flow is recorded; it never holds a zero |

## Cleaning log

TODO, with stage 2: one line per decision, with the number of rows it affected.
