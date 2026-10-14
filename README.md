<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="MemTrial" width="60%">
  </picture>
</p>

<h1 align="center">MemTrial: Learning When to Trust Memory in LLM Portfolio Agents</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.11732"><img src="https://img.shields.io/badge/arXiv-2610.11732-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&amp;logoColor=white" alt="Python 3.10+">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Reproduce-offline%2C%20no%20API%20key-2E8B57.svg" alt="Offline reproduction, no API key"></a>
  <a href="https://github.com/moderncavemann/MemTrial/stargazers"><img src="https://img.shields.io/github/stars/moderncavemann/MemTrial?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub stars"></a>
  <a href="https://github.com/moderncavemann/MemTrial/forks"><img src="https://img.shields.io/github/forks/moderncavemann/MemTrial?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub forks"></a>
  <a href="https://hitscounter.dev/history?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FMemTrial"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FMemTrial&amp;label=Views&amp;icon=graph-up&amp;color=%231f4e9e&amp;message=&amp;style=flat&amp;tz=Australia%2FSydney" alt="Page views: today / total"></a>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> | <a href="README.zh-CN.md">简体中文</a>
</p>

> **Authors:** Guanghao Wu<sup>\*</sup>, Zhuo Cai<sup>\*</sup>, Shoujin Wang<sup>†</sup><br>
> University of Technology Sydney<br>
> <sup>\*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author

MemTrial is an experience-learning LLM agent for portfolio management that **learns when to trust its memory**. It puts memory on trial: drafts of the same decision with and without an experience face the same market, so the outcome they share cancels in their difference. It acts on what it has learned only after its values have predicted unseen dates, and otherwise stays anchored at a conservative reference such as the equal-weight (1/N) portfolio.

- ⚖️ **Counterfactual credit:** values each experience by what it changes on the same date, not by the market move of that date.
- 🔍 **Trust test:** acts on its learned values only after they have predicted unseen dates.
- ⚓ **Anchored action:** otherwise stays near a conservative reference and learns online how far to move from it.

## 🧭 Why MemTrial?

LLM agents for portfolio management learn from experience: they write experiences, such as short lessons, into their memory and credit each experience with the outcome of the decisions that used it. In financial markets, this outcome mostly reflects the market move shared by all decisions on that date. An experience retrieved before a rally is credited with the rally, and one retrieved before a sell-off is blamed for it. The credit tracks the market rather than the experience, and these agents often do worse than simply holding the equal-weight (1/N) portfolio.

<p align="center">
  <img src="assets/overview.png" alt="Existing agents and MemTrial" width="80%">
</p>

<p align="center"><sub><strong>(a)</strong> Existing agents credit experiences with market-driven outcomes and always use these credits. <strong>(b)</strong> MemTrial measures what each experience changes on the same date and acts on its values only after they have predicted unseen dates; otherwise it stays near 1/N.</sub></p>

### 📉 Outcome Credit Tracks the Market

On every date, we generate a draft with each subset of the four retrieved experiences and compare two measures for each experience: its outcome credit, the mean utility of the drafts that used it, and its counterfactual contribution, that mean minus the mean utility of the drafts without it. Outcome credit rises and falls with the market almost in lockstep, but is nearly unrelated to what the experience changes.

| Benchmark | Outcome credit<br>vs. 1/N | Contribution<br>vs. 1/N | Outcome credit<br>vs. contribution | Dates with a<br>detectable effect |
| --- | ---: | ---: | ---: | ---: |
| PortBench-Full | **+0.91** | +0.04 | +0.11 | 3–4 / 56 |
| PortBench-Raw | **+0.92** | +0.03 | +0.04 | 2–4 / 56 |
| InvestorBench | **+0.95** | −0.07 | −0.03 | 126–127 / 149 |
| ClassAlloc | **+0.97** | +0.02 | +0.02 | 53–57 / 71 |

<p align="center"><sub>Spearman correlation over dates, mean over three seeds. The utility of 1/N over the same holding period is the proxy for the market.</sub></p>

### 🎯 Two Challenges

- **Measuring what an experience changes.** Every measurement is a costly, noisy LLM call on a market date that never comes back. A contribution measured on one date is mostly noise: on PortBench, the retrieved experiences change the agent's utility detectably on only 2–4 of 56 months.
- **Knowing whether a learned value will hold.** Even where experiences clearly change the decision, as on about 85% of InvestorBench days, the contributions learned from past days do not predict those of the following days. Existing agents always act on their learned scores and have no fallback for when these scores do not hold.

## ⚙️ How It Works

MemTrial learns when to trust its memory through a closed loop of four steps on every decision date. Steps 1–2 measure what each experience is worth; steps 3–4 decide whether to act on it.

<p align="center">
  <img src="assets/framework.png" alt="One decision date of MemTrial" width="100%">
</p>

<p align="center"><sub>One decision date of MemTrial. Before the outcome, it acts on its learned values only if they have predicted unseen dates; after the outcome, it updates the values and scores their predictions. Contributions compare drafts of one date, so their shared market move cancels.</sub></p>

1. **Counterfactual attribution.** Generates eight drafts of the decision with subsets of the four retrieved experiences chosen by a fractional factorial design, and estimates the contribution of each experience as its Banzhaf value, the average difference between the drafts with and without it. All drafts face the same market, so the outcome they share cancels.
2. **Value estimation.** Uses a hierarchical Bayesian model, as in small-area estimation, to pool the noisy contributions into values across dates and across experiences with similar content.
3. **Trust test.** Scores the predicted values against the contributions measured once the outcome is known, and trusts the values only when these out-of-sample forward scores are significantly positive.
4. **Anchored action.** If the values are trusted, deploys the draft with the highest predicted value; otherwise deploys a blend of the drafts anchored at a conservative reference, such as 1/N, and learns online how far to move from it.

The method is in [`memtrial/memtrial.py`](memtrial/memtrial.py). It was frozen before any InvestorBench decision (hash in `memtrial/MEMTRIAL_FROZEN.json`), so InvestorBench, ClassAlloc and the runs with other LLMs are held-out tests.

## 📊 Results

MemTrial is the best of 15 methods in PlantedMem, where the quality of every experience is known, and the best LLM-based method on PortBench and InvestorBench. There, no LLM-based method beats 1/N; MemTrial stays within 2.2% of it, whereas the best experience-learning agent falls 15–38% behind. Averaged over the five settings, its utility is 21.2% higher than that of the best experience-learning agent.

<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">PortBench-<br>Full<br><sub>%/month</sub></th>
      <th align="center">PortBench-<br>Raw<br><sub>%/month</sub></th>
      <th align="center">InvestorBench<br><sub>bp/day</sub></th>
      <th align="center">ClassAlloc<br><sub>%/month</sub></th>
      <th align="center">PlantedMem<br><sub>%/month</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="6"><i>Rule-based</i></td></tr>
    <tr><td>1/N</td><td align="center"><b>2.02</b></td><td align="center"><b>2.02</b></td><td align="center"><b>10.45</b></td><td align="center">0.623</td><td align="center">0.57</td></tr>
    <tr><td>Minimum variance</td><td align="center">0.25</td><td align="center">0.25</td><td align="center">5.53</td><td align="center">0.230</td><td align="center">−0.04</td></tr>
    <tr><td colspan="6"><i>LLM without experience learning</i></td></tr>
    <tr><td>Memory-free</td><td align="center">1.60</td><td align="center">1.07</td><td align="center">8.43</td><td align="center"><ins>0.657</ins></td><td align="center">0.76</td></tr>
    <tr><td>Self-consistency</td><td align="center">1.56</td><td align="center">1.17</td><td align="center">7.89</td><td align="center"><b>0.686</b></td><td align="center">0.79</td></tr>
    <tr><td>Similarity retrieval (top-2)</td><td align="center">1.43</td><td align="center">1.24</td><td align="center">8.67</td><td align="center">0.649</td><td align="center">0.72</td></tr>
    <tr><td>Similarity retrieval (top-4)</td><td align="center">1.70</td><td align="center">1.05</td><td align="center">8.12</td><td align="center">0.643</td><td align="center">0.72</td></tr>
    <tr><td colspan="6"><i>Experience-learning agents</i></td></tr>
    <tr><td>FinMem</td><td align="center">1.63</td><td align="center">1.26</td><td align="center">8.47<sup>†</sup></td><td align="center">0.594<sup>†</sup></td><td align="center">0.75<sup>†</sup></td></tr>
    <tr><td>MemRL</td><td align="center">1.58</td><td align="center">1.05</td><td align="center">8.90<sup>†</sup></td><td align="center">0.633<sup>†</sup></td><td align="center">0.74<sup>†</sup></td></tr>
    <tr><td>Reflexion</td><td align="center">1.40</td><td align="center">1.26</td><td align="center">7.29</td><td align="center">0.590</td><td align="center">0.78<sup>†</sup></td></tr>
    <tr><td>ExpeL</td><td align="center">1.53<sup>†</sup></td><td align="center">1.10<sup>†</sup></td><td align="center">8.93</td><td align="center">0.609</td><td align="center">0.77<sup>†</sup></td></tr>
    <tr><td colspan="6"><i>Credit-based and ensemble learners</i></td></tr>
    <tr><td>Uplift credit</td><td align="center">1.38<sup>†</sup></td><td align="center">1.20<sup>†</sup></td><td align="center">8.44<sup>†</sup></td><td align="center">0.645<sup>†</sup></td><td align="center">0.78<sup>†</sup></td></tr>
    <tr><td>Counterfactual selection</td><td align="center">1.49</td><td align="center">1.30</td><td align="center">8.54</td><td align="center">0.650</td><td align="center"><ins>0.85</ins></td></tr>
    <tr><td>Draft averaging</td><td align="center">1.62</td><td align="center">1.19</td><td align="center">8.69</td><td align="center">0.651</td><td align="center">0.75</td></tr>
    <tr><td>Hedge</td><td align="center">1.73</td><td align="center">1.59</td><td align="center">8.97</td><td align="center">0.634</td><td align="center">0.75<sup>†</sup></td></tr>
    <tr><td colspan="6"><i>Ours</i></td></tr>
    <tr><td><b>MemTrial</b></td><td align="center"><ins>1.98</ins></td><td align="center"><ins>2.01</ins></td><td align="center"><ins>10.40</ins></td><td align="center">0.613</td><td align="center"><b>0.87</b><sup>*</sup></td></tr>
    <tr><td><i>Improv.</i> (%)</td><td align="center">+21.3</td><td align="center">+59.0</td><td align="center">+16.4</td><td align="center">−3.2</td><td align="center">+12.6</td></tr>
    <tr><td>Trust rate (%)</td><td align="center">0.0</td><td align="center">0.6</td><td align="center">0.3</td><td align="center">14.6</td><td align="center">33.7</td></tr>
  </tbody>
</table>

<p align="center"><sub>Utility per decision period (certainty-equivalent return; higher is better), averaged over dates and three investors; mean over seeds. PlantedMem: mean of the five core regimes. <b>Bold</b>: best; <ins>underlined</ins>: second best. †: memory-scoring rule re-implemented. *: significant improvement (p &lt; 0.05). <i>Improv.</i>: gain over the best experience-learning agent of each setting. Trust rate: share of decisions on which MemTrial acts on its learned values.</sub></p>

On ClassAlloc, the memory-free LLM beats 1/N, but no form of memory improves on it; this advantage lies in the months before the LLM's knowledge cutoff.

<p align="center">
  <a href="assets/wealth.png"><img src="assets/wealth.png" alt="Wealth relative to 1/N" width="100%"></a>
</p>

<p align="center"><sub>Wealth of the balanced investor (top, start = 1) and relative to 1/N for the conservative, balanced and aggressive investors (rows 2–4). For every investor, MemTrial stays close to 1/N, while the other LLM-based methods drift away, mostly below.</sub></p>

### 🔬 What MemTrial Learns

PlantedMem keeps real prices and real LLM drafts but plants experiences of known quality, so the learned values can be checked against the truth.

<table>
  <tr>
    <td width="59%" align="center" valign="top">
      <a href="assets/recovery.png"><img src="assets/recovery.png" alt="Value recovery in PlantedMem" width="100%"></a><br>
      <sub><strong>Does MemTrial learn what each experience is worth?</strong> Outcome credit does not learn which experiences are useful; counterfactual contributions do, and pooling them by content helps when content is informative.</sub>
    </td>
    <td width="41%" align="center" valign="top">
      <a href="assets/regime.png"><img src="assets/regime.png" alt="PlantedMem by regime" width="100%"></a><br>
      <sub><strong>When does MemTrial trust its values?</strong> It abstains when experiences carry no signal and acts on its values when they matter.</sub>
    </td>
  </tr>
</table>

### 🤖 Robustness to the LLM

We repeat InvestorBench and ClassAlloc with seven other LLMs (gpt-4.1-nano, gpt-5-mini, Gemini 2.5 Flash, Claude Haiku 4.5, Llama-3.3-70B, Qwen3-235B and DeepSeek-V3.1) and run InvestorBench at temperatures 0, 0.3 and 1.0. On InvestorBench, no LLM-based baseline beats 1/N in any of the eleven configurations, and MemTrial beats all of them in every configuration, by 4–33% over the best experience-learning agent. On ClassAlloc, MemTrial stays within 0.021 pp per month of 1/N with every LLM.

## 🧪 Benchmarks

| Benchmark | Setting | Directory |
| --- | --- | --- |
| **PortBench** | 99 assets in 6 classes, monthly decisions, 20 test months (2023–24). PortBench-Raw keeps the native prompt, which shows prices for 8 assets; PortBench-Full shows every eligible asset. | [`portbench/`](portbench/), [`portbench_prices/`](portbench_prices/), [`portbench_after_cutoff/`](portbench_after_cutoff/) |
| **InvestorBench** | 4 stocks and cash with daily news, daily decisions, 149 test days (Oct 2020–May 2021). | [`investorbench/`](investorbench/) |
| **ClassAlloc** | 5 asset classes built from PortBench's prices, monthly decisions, 71 test months (May 2020–Mar 2026), ending 21 months after the knowledge cutoff of gpt-4.1-mini. | [`classalloc_and_robustness/`](classalloc_and_robustness/) |
| **PlantedMem** | Semi-synthetic: real prices and real LLM drafts with 12 or 48 planted experiences of known quality, 100 test worlds. | [`plantedmem/`](plantedmem/) |

All LLM agents use gpt-4.1-mini at temperature 0.7. Utilities are averaged over PortBench's conservative, balanced and aggressive investors.

## 🚀 Quick Start

Python 3.10+ with numpy and pandas. No GPU and no API key: every number in the paper's tables is recomputed offline from the included LLM outputs.

```bash
git clone https://github.com/moderncavemann/MemTrial.git
cd MemTrial
pip install -r requirements.txt
bash reproduce.sh                  # about two hours on 4 cores
python3 tables/print_tables.py     # Tables 1, 3 and Figure 4 -> tables/PAPER_TABLES.md
python3 wealth/return_risk.py      # Table 4 -> wealth/RETURN_RISK.json
```

Every step of `reproduce.sh` can also be run on its own. Tables 1, 3 and 4 and Figure 4 are reproduced to the last reported digit. The [reproduction guide](REPRODUCE.md) lists where every result is written and how to run the LLMs yourself.

## 🗂️ Repository Layout

| Path | Purpose |
| --- | --- |
| [`memtrial/`](memtrial/) | MemTrial (`memtrial.py`), hyperparameter variants and analyses. |
| [`plantedmem/`](plantedmem/) | PlantedMem: environment with real prices and logged LLM drafts, every method and variant. |
| [`portbench/`](portbench/) | PortBench: exact settlement, investor mandates and every method; every logged draft in `data/`. |
| [`portbench_after_cutoff/`](portbench_after_cutoff/) | PortBench with the twelve decisions of 2025–2026. |
| [`portbench_prices/`](portbench_prices/) | PortBench's public price data, mandate projection and settlement. |
| [`investorbench/`](investorbench/) | InvestorBench: LLM runs, evaluation of every method and logged drafts. |
| [`classalloc_and_robustness/`](classalloc_and_robustness/) | ClassAlloc, other LLMs and temperatures, Self-consistency and look-ahead checks. |
| [`ablation/`](ablation/) | Every ablation variant, and the LLM-based baselines with MemTrial's anchor. |
| [`tables/`](tables/), [`table1/`](table1/) | Mean and sd over seeds of every reported number; `print_tables.py`. |
| [`wealth/`](wealth/), [`figures/`](figures/), [`case_study/`](case_study/), [`prompts/`](prompts/) | Wealth paths, return and risk, figure data, ClassAlloc case study and prompt dump. |
| [`assets/`](assets/) | Logo and figures of this README. |

## 📚 Documentation

- Paper: [arXiv:2610.11732](https://arxiv.org/abs/2610.11732), DOI [10.48550/arXiv.2610.11732](https://doi.org/10.48550/arXiv.2610.11732).
- [Reproduction guide](REPRODUCE.md): every folder in detail, where each table and figure is written, the paid LLM runs and what is not included.
- [`reproduce.sh`](reproduce.sh): every step of the offline reproduction, in order.

## 🖋 Citation

If this repository is useful for you, please cite as:

```bibtex
@misc{wu2026memtrial,
      title={MemTrial: Learning When to Trust Memory in LLM Portfolio Agents},
      author={Guanghao Wu and Zhuo Cai and Shoujin Wang},
      year={2026},
      eprint={2610.11732},
      archivePrefix={arXiv},
      doi={10.48550/arXiv.2610.11732},
      url={https://arxiv.org/abs/2610.11732},
}
```

If you have any questions, please open an issue.
