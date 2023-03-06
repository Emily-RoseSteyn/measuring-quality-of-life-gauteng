# Beyond Poverty: Measuring Quality of Life from Satellite Images in Gauteng
Previous work has shown that asset wealth and consumption expenditure (recognised indicators of poverty) can be estimated from satellite images.

**Question**: Can other socioeconomic indicators be estimated from satellite images that represent a more holistic picture of quality of life?

- Focus on smaller area: Gauteng
- Start with satellite data that is lower res from Landsat
  - Then compare to higher res data from Planet
- First get asset wealth and income with nightlight proxy
- Consider other proxies

## Data Retrieval
- **Gauteng City Region Observatory Data**
  - Download GCRO dataset for 2021 from [Data First](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/874) portal.
  - Unzip folder
  - Rename the household DTA file to `gcro-2021.dta`
  - Copy DTA file to the [data](data) directory

Your data directory should now look like:
```
data
  gcro-2021.dta
```


[//]: # (TODO: Add years for dataset)