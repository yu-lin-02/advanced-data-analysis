# Data

<!-- Drafted with Claude (Anthropic). Asked for: a place to record where the data came from, its licence and how it was cleaned (Requirements 4.5 and 4.15), then filled in for the CEPII Gravity data when stage 1 was written, and the cleaning log when stage 2 was written (18 October 2026). Later asked for (4 November 2026, decision 24 of the build plan, "Rerun with zeros"): which run the cleaned table and the cleaning log describe. Later asked for (4 November 2026, after Yu's run of stage 2 with the zero flows and the freeze of decision 27): the cleaning log of the run with the zero flows, from its committed files and the counts of section 3 of the build plan, which stage 2 checks. Later asked for (5 November 2026, after the reproduction test of the run with the zero flows): the size and the SHA-256 fingerprint of `gravity_clean.csv` that stage 2 printed in that test, which Yu pasted from its log. -->

| File | Source | Downloaded on | Licence | In the repository? |
|---|---|---|---|---|
| `raw/Gravity_V202211.csv` | CEPII Gravity database, version V202211 ([page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8), [zip](https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip)) | Before 3 October 2026 (Yu's copy). Stage 1 prints the date of each new download. | Etalab Open Licence 2.0 | No: 1.25 GB, over GitHub's 100 MB limit. `code/01_get_data.ipynb` downloads it into `data/raw/`, which `.gitignore` leaves out (Requirement 4.15). |
| `gravity_extract.csv` | Cut from the file above by `code/01_get_data.ipynb` | | Etalab Open Licence 2.0 | Yes, 8.3 MB |
| `gravity_clean.csv` | Cleaned from the extract by `code/02_clean.ipynb` | | Etalab Open Licence 2.0 | Yes, 7.4 MB. The committed file is that of the run with the zero flows (decision 24 of the build plan), written by stage 2 in the Nuvolos workspace (commit `ea500aa`). The file of the first run, which dropped every pair with no recorded flow, is in commit `45d1c31`. |

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

`code/02_clean.ipynb` takes these steps, in this order, and writes the same counts to `results/cleaning_log.csv`. The counts are those of the run with the zero flows (decision 24 of the build plan), in `results/cleaning_log.csv` of commit `ea500aa`. Every count below is also an acceptance check in the notebook, against section 3 of `docs/build_plan.md`: a different number stops the run. The shares of trade are of the recorded trade, the total of the recorded flows left after step 3.

| Step | Rows | 2017 | 2019 |
|---|---|---|---|
| | Rows in the year | 63,504 | 63,504 |
| 1 | Dropped: origin and destination are the same country | 252 | 252 |
| 2 | Dropped: either country does not exist in that year | 8,262 | 8,262 |
| 3 | Set to zero: no trade flow is recorded, and both countries have a recorded flow that year | 18,068 | 18,832 |
| 3 | Dropped: no trade flow is recorded, and one of the countries has no recorded flow that year | 5,038 | 5,038 |
| 4 | Dropped: the GDP of either country is missing (distance is never missing) | 8,946 | 9,752 |
| 5 | Dropped: any of the five indicators is missing | 1,610 | 1,200 |
| | Usable rows | 39,396 | 39,000 |
| | Of which zero flows | 10,434 | 10,893 |

- **Step 3.** The flow is never zero in the CEPII file: a pair with no recorded trade is missing. It becomes 0 when both countries have a recorded flow that year, as origin or as destination. The rows dropped here are those of the 11 identifiers that have no recorded flow in either year (`ESH`, `FRO`, `GLP`, `GUF`, `LIE`, `MCO`, `MTQ`, `MYT`, `PRI`, `REU`, `VAT`), for which a missing flow says nothing about their trade.
- **Step 4.** GDP is missing for 21 countries in 2017 (`AIA`, `BES`, `CCK`, `COK`, `CXR`, `ERI`, `FLK`, `GIB`, `IOT`, `MSR`, `NFK`, `NIU`, `PCN`, `PRK`, `SHN`, `SPM`, `SSD`, `TKL`, `VEN`, `VGB`, `WLF`) and 23 in 2019 (the same, plus `SXM` and `SYR`). Their rows hold 0.34% of the recorded trade in 2017 and 0.26% in 2019. Of the rows dropped here, 6,578 in 2017 and 7,201 in 2019 are zero flows.
- **Step 5, two groups that do not overlap.**
  - `comlang_off` and `comcol` are missing in every row of `ASM`, `CUW`, `GUM` and, in 2017 only, `SXM`: 1,604 rows in 2017, 1,056 of them zero flows, and 1,194 in 2019, 738 of them zero. `contig` and `fta_wto` are never missing.
  - `col_dep_ever` is missing for China with Hong Kong, China with Taiwan, and Israel with Palestine, in both directions: 6 rows in each year, all with a recorded flow. The CEPII documentation (section 5.3) leaves these ties missing on purpose. They hold 2.47% of the recorded trade in 2017 and 2.43% in 2019, and the flow from China to Hong Kong is the fourth-largest of each year. They are dropped like every other row with a missing predictor (decision 10 of the build plan).
- **Kept.** Every recorded flow, however small: 1,562 usable rows in 2017 and 1,524 in 2019 have a positive flow below 1,000 USD (decision 3, no floor).
- **Then.** The five indicators are stored as integers (each holds only 0 and 1), the natural logs of the two GDPs and of the distance are added, and `pair_id` names the unordered pair. The usable rows form 19,698 unordered pairs in 2017 and 19,500 in 2019, each with both directions. Both directions are zero in 3,548 pairs in 2017 and 3,880 in 2019, and one direction is zero and the other positive in 3,338 and 3,133.

The first run dropped every row with no recorded flow at step 3, and its counts are those of `results/cleaning_log.csv` in commit `45d1c31`: 23,106 rows in 2017 and 23,870 in 2019 dropped at step 3, then 2,368 and 2,551 at step 4 and 554 and 462 at step 5, leaving 28,962 and 28,107 usable rows. These are the rows with a positive flow in the run with the zero flows.

`gravity_clean.csv` keeps 13 columns: `year`, `country_id_o`, `country_id_d`, `pair_id`, `tradeflow_baci` (in levels, thousand current USD) and the eight predictors `ln_gdp_o`, `ln_gdp_d`, `ln_dist`, `contig`, `comlang_off`, `col_dep_ever`, `comcol`, `fta_wto`. It has one row per usable row of the two years, 78,396 in all, and the notebook checks the count of each year. The notebook also prints the size and the SHA-256 fingerprint of the file it saves. In the reproduction test of 5 November 2026, a fresh clone of `ba08898` run in the Nuvolos workspace, it printed 7.4 MB and `e5cc432cebfafb3685988b94f2a41552d803e935540ed31edf41564bc289147c`, and the file it wrote was the same as the committed one.
