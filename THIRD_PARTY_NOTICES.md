# Third-party research and code provenance

MSPC's patent-specific state representation, deterministic conflict ordering,
three-objective Pareto weight search, validation feedback, history buffer, and
transactional rollback are implemented in this repository from the supplied
specification.

The optional `gradient_scaling="preserve"` and `descent_guard="mgda"` engineering
variants were built after studying these official open-source implementations:

| Project | Upstream | Used in this repository | License |
|---|---|---|---|
| PCGrad | <https://github.com/tianheyu927/PCGrad> | Projection against conflicting task gradients; comparison of raw-target versus reference normalization | MIT, Copyright (c) 2020 Tianhe Yu |
| CAGrad | <https://github.com/Cranial-XIX/CAGrad> | Official NYUv2 implementation's scale-preserving PCGrad pattern and common-descent motivation | MIT, Copyright (c) 2022 Bo Liu |
| MGDA | <https://github.com/isl-org/MultiObjectiveOptimization> | Minimum-norm convex-combination solver adapted in `mspc/hybrid.py` | MIT, Copyright (c) 2018 Ozan Sener |
| ParetoMTL | <https://github.com/Xi-L/ParetoMTL> | Reference comparison for Pareto preference vectors and simplex weight normalization; no source copied | MIT, Copyright (c) 2019 Xi Lin |
| FAMO | <https://github.com/Cranial-XIX/FAMO> | Reference comparison for history-aware relative loss improvement; no source copied into the patent risk formula | MIT, Copyright (c) 2023 Bo Liu |
| LibMTL | <https://github.com/median-research-group/LibMTL> | Framework adapter contract and baseline implementations | MIT, Copyright (c) LibMTL contributors |

The MGDA adaptation retains its upstream attribution. The MIT license applicable
to the listed MIT-licensed components is reproduced below.

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
