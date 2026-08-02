# pipetransport

`pipetransport` computes the water quality delivered at the far end of a branched pipe network — a drinking water distribution system — from the quality of the produced water, the pipe dimensions, and the demand metered at each delivery point. It also runs the other way: from quality measured at a few delivery points, back to the quality that must have left the treatment plant. Water age and chlorine residual come out of the same machinery. All without a hydraulic solver!

|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Testing of source code | <a href="https://github.com/gwtransport/pipetransport/actions/workflows/site.yml"><img alt="CI" src="https://github.com/gwtransport/pipetransport/actions/workflows/site.yml/badge.svg?branch=main" width="66" height="20"></a> <a href="https://gwtransport.github.io/pipetransport/htmlcov/"><img alt="Test Coverage" src="https://gwtransport.github.io/pipetransport/coverage-badge.svg" width="114" height="20"></a> <a href="https://github.com/gwtransport/pipetransport/actions/workflows/linting.yml"><img alt="Linting" src="https://github.com/gwtransport/pipetransport/actions/workflows/linting.yml/badge.svg?branch=main" width="115" height="20"></a> <a href="https://github.com/gwtransport/pipetransport/actions/workflows/release.yml"><img alt="Build and release package" src="https://github.com/gwtransport/pipetransport/actions/workflows/release.yml/badge.svg?branch=main" width="222" height="20"></a> |
| Testing of examples    | <a href="https://gwtransport.github.io/pipetransport/htmlcov_examples/"><img alt="Example Coverage" src="https://gwtransport.github.io/pipetransport/coverage_examples-badge.svg" width="114" height="20"></a>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Package                | <a href="https://pypi.org/project/pipetransport/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pipetransport.svg?logo=python&label=Python&logoColor=gold" width="215" height="20"></a> <a href="https://pypi.org/project/pipetransport/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pipetransport.svg?logo=pypi&label=PyPI&logoColor=gold" width="105" height="20"></a> <a href="https://github.com/gwtransport/pipetransport/compare/"><img alt="GitHub commits since latest release" src="https://img.shields.io/github/commits-since/gwtransport/pipetransport/latest?logo=github&logoColor=lightgrey" width="163" height="20"></a>                                                                                                                                                                                                                                            |

## What you can do

- **Trace a contamination event** from the plant to every delivery point, with arrival times that follow the real demand pattern
- **Model chlorine residual** with separate bulk and wall decay, so thin service lines lose residual faster than trunk mains — as they do
- **Map water age** across the network and watch it swing over the day
- **Predict the temperature at the tap** from the weather and what the pipes are buried under, including the heat the network itself stores in the soil
- **Reconstruct the produced water quality** from measurements at a handful of delivery points
- **Size a monitoring campaign** by asking which part of the production history each sampling point actually constrains

## The idea

A branched network has one path from the plant to each delivery point, and a split does not change concentration. So the delivered quality is the produced quality, delayed — and the delay is set by how much water has to be pushed through each pipe on the way.

Inside a pipe of water volume `V` carrying flow `Q(t)`, a parcel entering at time `s` leaves when it has displaced exactly `V`. Chain that pipe by pipe and you have the arrival time at any node, exactly, for any demand pattern. Nothing has to be steady.

When the demands _do_ move in proportion, the chain collapses to a single number per path: `sum(V_i / f_i)`, with `f_i` the fraction of production that segment `i` carries. A shared trunk main enters every downstream path in full rather than being divided among them. `pipetransport` does not need that assumption, but it reproduces it exactly when it holds.

## Installation

```bash
pip install pipetransport
```

## Forward: what arrives at the taps

```python
import numpy as np
import pandas as pd

from pipetransport.examples import example_demand, example_network
from pipetransport.transport import source_to_endmember

network = example_network()  # plant -> trunk main -> two district mains -> four taps
tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")  # n+1 edges for n bins
demand = example_demand(tedges=tedges, network=network)  # m3/day, one column per tap

# A three-hour contamination event leaves the plant on 2 June, 06:00-09:00
cin = np.zeros(len(tedges) - 1)
cin[30:33] = 1.0

cout = source_to_endmember(
    cin=cin,
    flow=demand,
    tedges=tedges,
    cout_tedges=tedges,
    network=network,
)  # (4 taps, 168 hours), same units as cin

for tap, series in zip(network.endmembers, cout, strict=True):
    peak = int(np.nanargmax(series))
    print(f"{tap}: peak {series[peak]:.2f} at {tedges[peak]}")
```

Only the demand at the taps is needed. Every internal pipe flow follows from mass conservation, and the split is recomputed at every time step — no proportional-demand assumption.

## Reverse: what left the plant

```python
import numpy as np
import pandas as pd

from pipetransport.examples import example_demand, example_network
from pipetransport.transport import endmember_to_source

network = example_network()
tedges = pd.date_range("2025-06-01", "2025-06-15", freq="h")
demand = example_demand(tedges=tedges, network=network)

# Hourly grab samples at two taps; NaN wherever nothing was sampled
measured = pd.DataFrame(
    {"T1": np.full(len(tedges) - 1, 0.6), "T4": np.full(len(tedges) - 1, 0.4)},
    index=tedges[:-1],
)

cin = endmember_to_source(
    cout=measured,
    flow=demand,
    tedges=tedges,
    cout_tedges=tedges,
    network=network,
    nodes=["T1", "T4"],
    regularization_strength=1e-4,  # ~ (noise / signal)^2
)  # (336,), NaN where no measurement constrains the bin
print(f"reconstructed {np.isfinite(cin).sum()} of {len(cin)} production bins")
```

Each sampling point constrains a different — and moving — window of the production history, so several of them together pin down more than any one alone.

## Chlorine residual and water age

```python
import numpy as np
import pandas as pd

from pipetransport.examples import example_demand, example_network
from pipetransport.logremoval import segment_decay_rate
from pipetransport.residence_time import full
from pipetransport.transport import source_to_endmember

network = example_network()
tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
demand = example_demand(tedges=tedges, network=network)

# Bulk decay plus a wall reaction that scales with 4 / diameter
decay = segment_decay_rate(network=network, bulk_decay_rate=0.3, wall_decay_rate=0.02)

residual = source_to_endmember(
    cin=np.full(len(tedges) - 1, 1.0),  # dosed to 1 mg/L at the plant
    flow=demand,
    tedges=tedges,
    cout_tedges=tedges,
    network=network,
    decay_rate=decay,
)
age = full(flow=demand, tedges=tedges, network=network)  # days

for tap, res, hours in zip(network.endmembers, residual, age * 24, strict=True):
    print(f"{tap}: residual {np.nanmin(res):.2f}-{np.nanmax(res):.2f} mg/L, age up to {np.nanmax(hours):.1f} h")
```

## Scope

`pipetransport` models a **single source feeding a tree**: flow splits, never merges, and never reverses. That is what makes one source signal enough to describe the whole network. Loops, a second plant, storage tanks and flow reversals are outside it. Transport inside a pipe is plug flow, which is a good approximation for turbulent mains and an optimistic one for laminar service lines.

See [Assumptions](https://gwtransport.github.io/pipetransport/user_guide/assumptions.html) for the full list and what each one costs you.

## Examples and Documentation

- [Core concepts](https://gwtransport.github.io/pipetransport/user_guide/concepts.html) — the label coordinate, the arrival map, and why the flow-weighted average is exact
- [Assumptions](https://gwtransport.github.io/pipetransport/user_guide/assumptions.html) — when this package fits your network
- [Water quality in a distribution network](https://gwtransport.github.io/pipetransport/examples/01_Distribution_Network_Water_Quality.html) — the worked example notebook
- [Temperature at the tap](https://gwtransport.github.io/pipetransport/examples/02_Network_Temperature.html) — the heat pair on a real network, forward and reverse
- [How the heat model works, in pictures](https://gwtransport.github.io/pipetransport/examples/03_Heat_Exchange_Conceptual_Model.html) — the conceptual model in plain English, with cross-sections

Full documentation: [gwtransport.github.io/pipetransport](https://gwtransport.github.io/pipetransport/)

`pipetransport` is the distribution-network sibling of [gwtransport](https://github.com/gwtransport/gwtransport), which does the same kind of timeseries transport for groundwater.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0), a strong copyleft license that requires making source code of any modifications available. This ensures improvements remain available to the community.

| Permissions      | Conditions                     | Limitations |
| ---------------- | ------------------------------ | ----------- |
| ✓ Commercial use | ℹ Disclose source              | ✗ Liability |
| ✓ Distribution   | ℹ License and copyright notice | ✗ Warranty  |
| ✓ Modification   | ℹ Network use is distribution  |             |
| ✓ Patent use     | ℹ Same license                 |             |
| ✓ Private use    | ℹ State changes                |             |

For more details about this license, see the [full AGPL-3.0 license text](https://choosealicense.com/licenses/agpl-3.0/).
