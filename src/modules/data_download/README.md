# Data Download
- Get API key for Planet
- Earth engine later?

## Manual Data Retrieval
- **Gauteng City Region Observatory Data**
  - Download GCRO dataset for 2020–2021 from [Data First](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/874) portal.
  - Unzip folder
  - Rename the household DTA file to `gcro-2021.dta`
  - Copy DTA file to the [data/surveys](/data/surveys) directory
  - Repeat this for [2015–2016](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/595) and [2017–2018](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/766)

Your data directory should now look like:
```
data
  surveys
    gcro-2016.dta
    gcro-2018.dta
    gcro-2021.dta
```