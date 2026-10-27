# Ocean Modelling with ROMS: a hands-on class in Google Colab

Material for a hands-on class on regional ocean modelling with the **Regional Ocean Modeling System (ROMS)**, aimed at MSc and first-year PhD students.
Participants build their own Indian Ocean model inputs, compile and run ROMS, and analyse the output, all in **Google Colab**, with nothing to install on their own computers.

| File | What it is |
|---|---|
| [`Grid_IC_BC_Preparation_Steps.ipynb`](Grid_IC_BC_Preparation_Steps.ipynb) | **Notebook 1**: create and edit the model grid, initial conditions, boundary conditions and surface forcing |
| [`ROMS_class_2026.ipynb`](ROMS_class_2026.ipynb) | **Notebook 2**: compile ROMS, run it with the files from notebook 1, analyse the results |
| `presentation/Inside_an_Ocean_Model.pdf` | **Lecture slides** (41 slides, about 60 minutes): how an ocean model is built, fed and solved |

---

## How the pieces fit together

```mermaid
flowchart LR
    A[Notebook 1: Grid_IC_BC_Preparation_Steps.ipynb] -->|roms_grd, roms_ini, roms_frc, roms_bry| D[(Google Drive: ROMS_class/inputs)]
    D --> B[Notebook 2: ROMS_class_2026.ipynb]
    B --> C[ROMS output: maps, sections, heat budget]
```

Each Colab notebook runs on its own virtual machine, so the two notebooks share files through **Google Drive**.
Notebook 1 saves four files to `MyDrive/ROMS_class/inputs/`; notebook 2 copies them from there.

**Suggested schedule**

| Session | Content | Time |
|---|---|---|
| Lecture | Slides: *Inside an Ocean Model* | 60 min |
| Hands-on 1 | Notebook 1: grid, initial, boundary and forcing files | 30–45 min |
| Hands-on 2 | Notebook 2: compile, run, analyse | 45–60 min |

---

## Quick start for participants

1. Download both `.ipynb` files from this repository (open the file → **⋯** → **Download**).
2. Go to [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → choose **`Grid_IC_BC_Preparation_Steps.ipynb`**.
3. Run all cells from top to bottom (**Runtime → Run all**).
   * If Colab asks to **restart the session** after installing `roms-tools`, click *Restart* and continue from the next cell.
   * The last cell asks for permission to use your Google Drive. Allow it: this is where your files are saved.
4. Upload **`ROMS_class_2026.ipynb`** to Colab and run all cells from top to bottom.
   It finds your files in Drive automatically and runs ROMS with them.

> **No time for notebook 1?** In notebook 2, Step 1, set `INPUTS = 'tutorial'` to run with ready-made input files instead.

---

## Notebook 1: `Grid_IC_BC_Preparation_Steps.ipynb`

Builds every input file ROMS needs with the [**roms-tools**](https://roms-tools.readthedocs.io/en/latest/) Python package (pinned to version 5.2.2).
Runtime on Colab: about 1 minute after installation.

| Step | What happens | roms-tools class | Source data |
|---|---|---|---|
| Setup | install `roms-tools`, clone the data repository, common settings (one `model_reference_date` for all files) | | [`prasannakanti/Data_Grid`](https://github.com/prasannakanti/Data_Grid) |
| Grid | 91 × 61 interior points (~100 km), 50 vertical levels, SRTM15 bathymetry, coastline mask, narrow channels closed | `Grid` | SRTM15+ (pre-averaged to 1°) |
| Mask editor | **interactive**: click or drag to switch cells between land and ocean, then *Apply to grid* | | |
| Diagnostics | mask, bathymetry, vertical stretching, vertical sections, topography steepness (rx0) | | |
| Initial conditions | T, S, u, v, ζ on 16 Jan 1993, after filling sea-floor gaps in GLORYS (see below) | `InitialConditions` | GLORYS12 monthly mean |
| Boundary conditions | south and east boundaries, with fill across the islands of Indonesia | `BoundaryForcing` | GLORYS12 monthly mean |
| Surface forcing | wind, air temperature and humidity, radiation, rain; Dec 1992 – Jan 1994 | `SurfaceForcing` | ERA5 monthly means |
| Convert & save | converts to Rutgers-ROMS format and saves to Google Drive | | |

**Output** (in `MyDrive/ROMS_class/inputs/`):

| File | Content |
|---|---|
| `roms_grd.nc` | grid: bathymetry, land–sea mask, metrics, Coriolis, ρ/u/v/ψ coordinates |
| `roms_ini.nc` | initial state + the s-coordinate parameters (read automatically by notebook 2) |
| `roms_frc.nc` | monthly surface forcing on the model grid, including air pressure |
| `roms_bry.nc` | open-boundary data (made as an exercise; **not used**, the class run has closed boundaries) |

**Things this notebook handles that are easy to get wrong**

* **Sea-floor gaps.** The smoothed 1° model is much deeper than GLORYS over small banks (Seychelles Bank, Saya de Malha). Without filling those gaps at the same depth, warm surface water ends up at 1,000 m and ROMS blows up within days. A check cell reports the largest deep temperature jump.
* **One time reference.** All files use the same `model_reference_date`; mixing reference dates shifts the forcing relative to the start of the run.
* **ERA5 monthly means.** roms-tools expects *hourly* accumulations, while monthly means hold *daily* accumulations of radiation and rain, so these are divided by 24. Monthly records are re-stamped from the 1st to the 15th of each month.
* **roms-tools writes UCLA-ROMS files.** The final step converts them for Rutgers ROMS: ψ-points and metric terms in the grid, times in seconds/days since 1970, variable names matching `varinfo.yaml`, and air pressure (which roms-tools does not provide).

**ERA5 data.** By default the notebook builds a practice file from `ERA5.nc` in the data repository, so it runs without an account.
For your own year, download *ERA5 monthly averaged reanalysis on single levels* from the [Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download):
10 m u/v wind, 2 m temperature, 2 m dewpoint, surface net solar radiation, surface thermal radiation downwards, total precipitation, sea surface temperature.
Name the file `ERA5_monthly_raw.nc` and set `ERA5_OPTION = 'A'` (instructions in the notebook's exercise).

---

## Notebook 2: `ROMS_class_2026.ipynb`

Compiles and runs **Rutgers ROMS** ([myroms.org](https://www.myroms.org)), then analyses the output with `xarray`, `numpy` and `cartopy`.
Runtime on Colab: about 2 minutes to compile, 3–5 minutes for 30 model days.

| Step | What happens |
|---|---|
| 1 Install | NetCDF-Fortran, OpenMPI and cartopy; downloads the tutorial configuration; clones ROMS at a fixed commit (`f8a3aaa2`, Aug 2025) |
| 1 Inputs | `INPUTS = 'mine'` copies your files from Google Drive; `'tutorial'` uses the ready-made files |
| 2 Inspect | maps of bathymetry, initial SST and forcing; explains two common forcing traps (downward vs net longwave, monthly-mean time stamps) |
| 3 Configure | writes the physics switches (`io_1deg.h`); sets run length, viscosity, output frequency, grid size and s-coordinates from **your** files in `roms_io_1deg.in`; sets the start time; **closes all four boundaries** and checks them |
| 4 Compile | `build_roms.sh -j 2`, log written to `build.log` |
| 5 Run | 30-day run, log written to `roms.log`; clear error messages if ROMS stops; plots of kinetic energy and maximum speed |
| 6 Save | download or copy output to Google Drive |
| 7 Analyse | SST and SST change, surface currents, temperature section with locator map, KPP mixed-layer depth, surface heat budget, spin-up time series, comparison with GLORYS |
| 8 Exercises | viscosity, vertical mixing, the longwave trap, restarting a run, more sections |

**Model configuration**

| Setting | Value |
|---|---|
| Domain | Indian Ocean, ~34°E–116°E, 27°S–27°N, about 1° (≈100 km) |
| Vertical grid | 50 terrain-following levels (Vtransform 2, Vstretching 4, θs = 5, θb = 0.7, hc = 300 m) |
| Boundaries | closed walls on all four sides |
| Time step | 3000 s baroclinic, 50 s barotropic (NDTFAST = 60) |
| Vertical mixing | KPP (`LMD_MIXING`) |
| Horizontal mixing | Laplacian viscosity 10⁴ m²/s, diffusivity 10³ m²/s (along geopotentials) |
| Advection | 3rd-order upstream (horizontal), 4th-order centred (vertical) |
| Surface fluxes | bulk formulae (`BULK_FLUXES`, `LONGWAVE_OUT`), penetrating shortwave |
| Start / length | 16 Jan 1993, 30 days (change `RUN_DAYS`) |
| Output | 5-day averages, 10-day snapshots, restart file |

---

## Presentation: *Inside an Ocean Model*

41 slides for a 60-minute lecture, with diagrams on almost every slide and speaker notes on each one.
Many examples come from real problems met while preparing this class.

| Part | Topics |
|---|---|
| Introduction | what a model is, the hydrostatic primitive equations |
| 1 Build & run | the compile pipeline, CPP switches, anatomy of a run |
| 2 Inputs | grid file, initial conditions, bulk-formula surface forcing, open-boundary conditions |
| 3 Grids | Arakawa A/B/C grids, the ROMS C-grid cell, z / terrain-following / isopycnal coordinates, s-coordinate stretching, pressure-gradient error |
| 4 Numerics | the finite-volume method, a linear 1-D advection example with worked numbers, upwind vs Lax–Wendroff, CFL and mode splitting, the temperature tendency equation term by term, the 3-D tracer cell, ROMS time stepping, advection schemes |
| 5 Parameterizations | scales of ocean motion, KPP vertical mixing, Gent–McWilliams mesoscale eddies, Fox-Kemper submesoscale eddies, viscosity, what to parameterize at each resolution |
| 6 Wrap-up | one ROMS time step in code order, eight sanity checks, references |

---

## Requirements

* A Google account (for Colab and Google Drive)
* Nothing to install locally. Notebooks install `roms-tools`, `cartopy`, gfortran, NetCDF-Fortran and OpenMPI on the Colab machine.
* Optional: a free [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) account for your own ERA5 download, and a [Copernicus Marine](https://marine.copernicus.eu) account for more GLORYS months.

---

## Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `FileNotFoundError: ['roms_grd.nc', …] not found` (notebook 2) | Notebook 1 was not run to the end, or Drive was not mounted. Run its last cell, or set `INPUTS = 'tutorial'`. |
| `NETCDF_GET_SATT - error while reading attribute: units` | A file was written with NetCDF-4 string attributes. The notebooks write NetCDF-3 to avoid this; re-run the conversion cell. |
| `Blowing-up: Saving latest model state` | Usually an input problem: check the deep temperature jump in notebook 1, the maximum speed in the log, and the grid's rx0. |
| Max speed of several m/s on day 1 | Horizontal viscosity too small for the grid spacing (`VISC2`). |
| Colab asks to restart after installing roms-tools | Normal: restart, then continue from the next cell. |
| Run disconnected | Colab sessions time out; reduce `RUN_DAYS`, or save the restart file to Drive. |

---

## Data and software credits

* **ROMS**: Rutgers ROMS, [myroms.org](https://www.myroms.org), [github.com/myroms/roms](https://github.com/myroms/roms)
* **roms-tools**: [CWorthy-ocean/roms-tools](https://github.com/CWorthy-ocean/roms-tools), [documentation](https://roms-tools.readthedocs.io/en/latest/)
* **Input data repository**: [prasannakanti/Data_Grid](https://github.com/prasannakanti/Data_Grid) (SRTM15 subset, GLORYS January 1993, ERA5 monthly means)
* **Tutorial configuration**: [Rohithocean/physical_oceanography](https://github.com/Rohithocean/physical_oceanography) (`ROMS_tutorial.zip`)
* **GLORYS12**: Copernicus Marine Service, Lellouche et al. (2021), *Front. Earth Sci.* 9, 698876
* **ERA5**: Copernicus Climate Change Service, Hersbach et al. (2020), *Q. J. R. Meteorol. Soc.* 146, 1999–2049
* **SRTM15+**: Tozer et al. (2019), *Earth Space Sci.* 6, 1847–1864

**Key references for the lecture:** Shchepetkin & McWilliams (2005), *Ocean Modelling* 9, 347–404 · Haidvogel et al. (2008), *J. Comput. Phys.* 227, 3595–3624 · Large, McWilliams & Doney (1994), *Rev. Geophys.* 32, 363–403 · Gent & McWilliams (1990), *J. Phys. Oceanogr.* 20, 150–155 · Fox-Kemper, Ferrari & Hallberg (2008), *J. Phys. Oceanogr.* 38, 1145–1165. The full list is on the last two slides.

---

## Authors and licence

* Class material: [Rohith B]
* Contact: [rohith.b@incois.gov.in]
* Licence: [CC BY 4.0 for the slides and notebooks]
