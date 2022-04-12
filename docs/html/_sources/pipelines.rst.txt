
Pipelines
==================


For Galaxies
------------------

The following pipelines are available for processing SDSS objects
classified as `GALAXY`.

**One-2-One Ensemble**

Infer redshift, stellar mass, subclass and gz2 simplified classification from
RGB image, FITS, spectra, selected spectra, bands and WISE data using an ensemble
of all the single input/output models available, described
in the following table.

+---------------------+------------------------------------------------+
| Output              | Ensemble                                       |
+=====================+================================================+
| :code:`redshift`    | :code:`[i2r, f2r, s2r, ss2r, b2r, w2r]`        |
+---------------------+------------------------------------------------+
| :code:`smass`       | :code:`[i2sm, f2sm, s2sm, ss2sm, b2sm, w2sm]`  |
+---------------------+------------------------------------------------+
| :code:`sublcass`    | :code:`[i2s, f2s, s2s, ss2s, b2s, w2s]`        |
+---------------------+------------------------------------------------+
| :code:`gz2c`        | :code:`[i2g, f2g, s2g, ss2g, b2g, w2g]`        |
+---------------------+------------------------------------------------+

**Cherry-Picked Ensemble**

Infer redshift, stellar mass, subclass and gz2 simplified classification from
RGB image, FITS, spectra, selected spectra, bands and WISE data using an ensemble
of hand picked models, described in the following table.

+---------------------+------------------------------------------------+
| Output              | Ensemble                                       |
+=====================+================================================+
| :code:`redshift`    | :code:`[f2r, s2r, ss2r, iFsSSbW2r]`            |
+---------------------+------------------------------------------------+
| :code:`smass`       | :code:`[f2sm, b2sm, w2sm, iFsSSbW2sm]`         |
+---------------------+------------------------------------------------+
| :code:`sublcass`    | :code:`[iFsSSbW2s]`                            |
+---------------------+------------------------------------------------+
| :code:`gz2c`        | :code:`[i2g, f2g, iFsSSbW2g]`                  |
+---------------------+------------------------------------------------+
