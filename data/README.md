# Data Download

- Get API key for Planet
- Earth engine later?

## Surveys: Gauteng City Region Observatory Data

- **NB! This is a manual data download step. Data First requires an explanation of use of their data before one can
  download it.**
- Steps to follow:
    - Register and login to the [Data First](https://www.datafirst.uct.ac.za/dataportal/index.php/auth/register) portal.
    - Download GCRO dataset for [2020–2021](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/874) the
      portal.
    - Unzip folder
    - Rename the household DTA file to `gcro-2021.dta`
    - Copy DTA file to the [data/surveys](/data/surveys) directory
    - Repeat this for [2015–2016](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/595)
      and [2017–2018](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/766)

Your data directory should now look like:

```
data
  surveys
    gcro-2016.dta
    gcro-2018.dta
    gcro-2021.dta
```

The GCRO 2017/18 and 2020/21 survey methods to calculate the Quality of Life index were modified. For this
reason an additional data file is included in the
repo - [gcro-2016-additional-data.sav](./surveys/gcro-2016-additional-data.sav). Make sure to run `dvc pull` if you do
not have hooks initialised. Then you can proceed with the `gcro-processing` step which will merge this data into the
gcro-2016 dataset.

## Shapefiles

The shapefiles have been added to the DVC remote store. Make sure that you have setup DVC as documented in the
root [readme](/README.md).

Now, when you run `git pull` on `develop`, you should find the shapefiles in the [data directory](/data/shapefiles).

#### Manual Download

If you do not have access to the DVC remote store, or you are setting up an independent remote store, you can manually
download this data from the South African [Municipal Demarcation Board](https://www.demarcation.org.za/).

* Download [2020 shapefile](https://www.arcgis.com/sharing/rest/content/items/e0223a825ea2481fa72220ad3204276b/data).
* Unzip and rename folder to 2020 and rename all child shapefiles to `2020.<ext>`.
* Place the entire folder in [data/shapefiles](/data/shapefiles).

2011 and 2016 ward boundaries are provided as geodatabase files, so these need to be converted to shapefiles.

* Download the [2016](https://www.arcgis.com/sharing/rest/content/items/cfddb54aab5f4d62b2144d80d49b3fdb/data) ward
  boundaries.
* Use a tool
  like [ArcMap](https://desktop.arcgis.com/en/arcmap/latest/extensions/production-mapping/converting-a-geodatabase-to-shapefiles.htm), [QGIS](https://gis.stackexchange.com/questions/108006/converting-data-from-gdb-into-shapefile-without-arcmap)
  or similar to import the geodatabase.
* Save as shapefile.
* Rename exported shapefile directory to 2016.
* Place the entire folder in [data/shapefiles](/data/shapefiles)
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

## Satellite Images

### Planet Access

This work was carried out using
the [Planet Education & Research Programme](https://www.planet.com/industries/education-and-research/). One can apply
via [their website](https://www.planet.com/industries/education-and-research/#apply-now) for a license with basic
access. This can take up to 2 weeks to obtain.

In order to download images from Planet, you need to find your personal API key
under [user settings](https://www.planet.com/account/#/user-settings). Copy the API key and paste it in
the [.env](../.env). Your .env file should now at minimum look like:

```
PLANET_API_KEY=PLAK**********
```

### Actual Download

If you have access to the DVC remote store, you can execute the `data_download` step in the DVC pipeline without any
additional work.
This step uses [basemap quad](https://developers.planet.com/docs/basemaps/#basemap-quads) identifiers stored
in [basemap-metadata.json](basemap-metadata/basemap-metadata.json) to download quads from Planet's API.
The process for creating this file is described below.

#### Manual Download

If you do not have access to the DVC remote store, you will need to do the following before you can execute
the `data_download` step.

First, run the following steps in the DVC pipeline:

* shapefile_processing
* gcro_processing

The outputs of these two steps include a gauteng boundary geojson file and date information about the GCRO surveys.

We can now use [Planet's basemap viewer](https://www.planet.com/basemaps) to do the following:

* Identify the mean date of the survey outputted from the `gcro_processing` step in the `gcro-date-info.json` file
* Search for the monthly MOSAIC that corresponds to the mean date.
    * For example, in the case of the 2021 GCRO survey, the mean date is 2021-01-12. The corresponding monthly mosaic on
      Planet is January 2021 (`global_monthly_2021_01_mosaic`).
* Find the relevant area of interest, by uploading the corresponding Gauteng boundary geojson generated from
  the `shapefile_processing` step.
* A selected location should be visible with the corresponding basemap quads.
* Copy the quad ids and create a `basemap-metadata.json` file with the following structure:

```json
{
  "[year]": {
    "mosaic_name": "[mosaic name from planet]",
    "basemap_quad_ids": [
      "copied quad ids wrapped in double quotations..."
    ]
  }
}
```

**NB!** Planet paginates its basemap quad id. Ensure you "Load More" until there are no more quads to load. There should
be ~91 quads for Gauteng.

In the future, the step for searching and selecting basemap quads will potentially be automated using Planet's API so
that one does not have to get the ids from the web interface.

## Custom Data

Some custom data was manually created. This can be retrieved from the DVC data store by running `dvc fetch` or
`dvc pull`. You should then see data populated under [custom](custom)

#### Custom Training Wards

A custom CSV file exists for selecting specific wards manually during data splitting for
training/testing [train_wards.csv](custom/train_wards.csv). See more in
the [select_data](../src/modules/7_data_splitting/select_data.py) function.
