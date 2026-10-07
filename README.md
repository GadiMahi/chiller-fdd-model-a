# Model A — Topology-aware chiller-plant fault isolation & "phantom kW/TR" costing

Finds **which piece of equipment is actually faulty** in a central chiller plant (instead of flagging every
sensor that looks odd downstream of it), names the **fault type**, and puts a **₹/day price** on it so
maintenance is ranked by money lost.

**Data:** LBNL Fault Detection & Diagnostics Data Sets — *Simulated chiller plant*
(Granderson et al., LBNL/PNNL 2022, DOI [10.25984/1881324](https://dx.doi.org/10.25984/1881324)):
77 points at 1-min for a year, 7 fault types at several severities (21 faulty runs + 1 fault-free run),
plus a Brick-schema `.ttl` model of the plant.

## Files
| File | What it is |
|---|---|
| `Model_A_Chiller_FDD_Phantom_Cost.ipynb` | The whole pipeline, with justification and results. Saved outputs come from the **stand-in data** (see below). |
| `lbnl_surrogate.py` | Physics-based stand-in that reproduces the LBNL point names, file names and fault list. Used only when the real CSVs are not attached, so the notebook can be tested end to end. |
| `sample_output_ticket_list_SURROGATE.csv` | Example of the ranked maintenance-ticket output (stand-in data). |
| `tools/build_notebook.py` | Script that generates the notebook (edit here if you prefer plain Python). |
| `requirements.txt` | Python packages (all preinstalled on Kaggle except possibly `rdflib`). |

## Run on Kaggle
1. **Get the data:** download *Simulated chiller plant data sets.zip* and the `.ttl` from
   <https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/> (short contributor form).
2. **Upload it to Kaggle:** *Datasets → New Dataset*, add the zip and the `.ttl`, keep it private.
3. **Open the notebook on Kaggle:** *Create → New Notebook → File → Import Notebook → GitHub*, paste `https://github.com/GadiMahi/chiller-fdd-model-a` and pick
   `Model_A_Chiller_FDD_Phantom_Cost.ipynb`.
4. In the notebook: *Add Input* → your LBNL dataset; *Settings → Internet → On*.
5. In the first code cell set your tariff in `TARIFF_INR_PER_KWH` (`REPO_URL` already points here).
6. *Run All*. The setup cell finds the CSVs anywhere under `/kaggle/input` and prints `MODE: REAL LBNL DATA`.
   CPU is enough; expect roughly 25–40 min.

Alternative: in any Kaggle notebook run `!git clone https://github.com/GadiMahi/chiller-fdd-model-a` and `%cd chiller-fdd-model-a`, then copy the cells in.

If the setup cell prints `MODE: SURROGATE`, it did not find the 22 CSVs and is using stand-in data —
**do not quote those numbers.**

## Method in one paragraph
The Brick graph gives which equipment feeds which. For every sensor, a fault-free model (ridge + gradient boosting)
predicts it from the other sensors on the same equipment and the sensors of the equipment feeding it, so a downstream
node reacting to an upstream fault stays "normal" while the faulty node breaks its own physics. Daily residual
z-scores flag anomalous nodes; the root cause is the anomalous node that no upstream node — or the pump/valve
feeding it an abnormal flow — explains. A classifier on the residual signature names the fault type and is tested
on severities it never saw. A weather-and-time baseline of plant power gives the counterfactual kW;
actual − counterfactual = phantom kW, converted to ₹/day and payback.
