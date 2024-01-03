# Data Download
- Get API key for Planet
- Earth engine later?

## Surveys: Gauteng City Region Observatory Data
- **NB! This is a manual data download step. Data First requires an explanation of use of their data before one can download it.**
- Steps to follow:
  - Register and login to the [Data First](https://www.datafirst.uct.ac.za/dataportal/index.php/auth/register) portal.
  - Download GCRO dataset for [2020–2021](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/874) the portal.
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

## Shapefiles
The shapefiles have been added to the DVC remote store. Make sure that you have setup DVC as documented in the root [readme](/README.md).

Now, when you run `git pull` on `develop`, you should find the shapefiles in the [data directory](/data/shapefiles).

#### Manual Download
If you do not have access to the DVC remote store, or you are setting up an independent remote store, you can manually download this data from the South African [Municipal Demarcation Board](https://www.demarcation.org.za/).
* Download [2020 shapefile](https://www.arcgis.com/sharing/rest/content/items/e0223a825ea2481fa72220ad3204276b/data).
* Unzip and rename folder to 2020 and rename all child shapefiles to `2020.<ext>`.
* Place entire folder in [data/shapefiles](/data/shapefiles).

2011 and 2016 ward boundaries are provided as geodatabase files, so these need to be converted to shapefiles.
* Download the [2016](https://www.arcgis.com/sharing/rest/content/items/cfddb54aab5f4d62b2144d80d49b3fdb/data) ward boundaries.
* Use a tool like [ArcMap](https://desktop.arcgis.com/en/arcmap/latest/extensions/production-mapping/converting-a-geodatabase-to-shapefiles.htm), [QGIS](https://gis.stackexchange.com/questions/108006/converting-data-from-gdb-into-shapefile-without-arcmap) or similar to import the geodatabase.
* Save as shapefile.
* Rename exported shapefile directory to 2016.
* Place entire folder in [data/shapefiles](/data/shapefiles)
* Repeat for [2011](https://www.arcgis.com/sharing/rest/content/items/12d2deb98816451ab7c4dc09cdfeee6b/data).

#### Expected Shapefile Directory
Your shapefile directory should now look like:
```
data
  shapefiles
    2011/
    2016/
    2020/
```
