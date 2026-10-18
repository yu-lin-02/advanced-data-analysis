# Data

<!-- Drafted with Claude (Anthropic). Asked for: a place to record where the data came from, its licence and how it was cleaned (Requirements 4.5 and 4.15), then filled in for the CEPII Gravity data when stage 1 was written, and the cleaning log when stage 2 was written (18 October 2026). -->

| File | Source | Downloaded on | Licence | In the repository? |
|---|---|---|---|---|
| `raw/Gravity_V202211.csv` | CEPII Gravity database, version V202211 ([page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8), [zip](https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip)) | Before 3 October 2026 (Yu's copy). Stage 1 prints the date of each new download. | Etalab Open Licence 2.0 | No: 1.25 GB, over GitHub's 100 MB limit. `code/01_get_data.ipynb` downloads it into `data/raw/`, which `.gitignore` leaves out (Requirement 4.15). |
| `gravity_extract.csv` | Cut from the file above by `code/01_get_data.ipynb` | | Etalab Open Licence 2.0 | Yes, 8.3 MB |
| `gravity_clean.csv` | Cleaned from the extract by `code/02_clean.ipynb` | | Etalab Open Licence 2.0 | Yes, 5.4 MB, once stage 2 has been run in the Nuvolos workspace. TODO: commit it from that run. |

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

`code/02_clean.ipynb` takes these steps, in this order, and writes the same counts to `results/cleaning_log.csv`. Every count below is also an acceptance check in the notebook, against the counts read on Yu's copy of the CEPII file on 3 October 2026 (section 3 of `docs/build_plan.md`): a different number stops the run. The shares of trade are of the recorded trade, the total of the flows left after step 3.

| Step | Rows dropped because | 2017 | 2019 |
|---|---|---|---|
| | Rows in the year | 63,504 | 63,504 |
| 1 | Origin and destination are the same country | 252 | 252 |
| 2 | Either country does not exist in that year | 8,262 | 8,262 |
| 3 | No trade flow is recorded. The flow is never zero: a pair with no recorded trade is missing | 23,106 | 23,870 |
| 4 | The GDP of either country is missing (distance is never missing) | 2,368 | 2,551 |
| 5 | Any of the five indicators is missing | 554 | 462 |
| | Usable rows | 28,962 | 28,107 |

- **Step 4.** GDP is missing for 21 countries in 2017 (`AIA`, `BES`, `CCK`, `COK`, `CXR`, `ERI`, `FLK`, `GIB`, `IOT`, `MSR`, `NFK`, `NIU`, `PCN`, `PRK`, `SHN`, `SPM`, `SSD`, `TKL`, `VEN`, `VGB`, `WLF`) and 23 in 2019 (the same, plus `SXM` and `SYR`). Their rows hold 0.34% of the recorded trade in 2017 and 0.26% in 2019.
- **Step 5, two groups that do not overlap.**
  - `comlang_off` and `comcol` are missing in every row of `ASM`, `CUW`, `GUM` and, in 2017 only, `SXM`: 548 rows in 2017 and 456 in 2019. `contig` and `fta_wto` are never missing.
  - `col_dep_ever` is missing for China with Hong Kong, China with Taiwan, and Israel with Palestine, in both directions: 6 rows in each year. The CEPII documentation (section 5.3) leaves these ties missing on purpose. They hold 2.47% of the recorded trade in 2017 and 2.43% in 2019, and the flow from China to Hong Kong is the fourth-largest of each year. They are dropped like every other row with a missing predictor (decision 10 of the build plan).
- **Kept.** Every recorded flow, however small: 1,562 usable rows in 2017 and 1,524 in 2019 have a flow below 1,000 USD (decision 3, no floor).
- **Then.** The five indicators are stored as integers (each holds only 0 and 1), the natural logs of the two GDPs and of the distance are added, and `pair_id` names the unordered pair. The usable rows form 16,150 unordered pairs in 2017 (12,812 with both directions) and 15,620 in 2019 (12,487).

`gravity_clean.csv` keeps 13 columns: `year`, `country_id_o`, `country_id_d`, `pair_id`, `tradeflow_baci` (in levels, thousand current USD) and the eight predictors `ln_gdp_o`, `ln_gdp_d`, `ln_dist`, `contig`, `comlang_off`, `col_dep_ever`, `comcol`, `fta_wto`. Run on the committed extract with the pinned packages (Python 3.12), stage 2 wrote 57,069 rows with SHA-256 `7c8eda4bd3c2e9295d38b902823af51abd46ca3757bed14cddf292d0c483a0c7`, which the notebook prints. TODO: confirm that the run in the Nuvolos workspace prints the same fingerprint.
