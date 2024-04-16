# CompMCTG Benchmark \& Meta-MCTG
## 1. How to use our CompMCTG Benchmark?
For each method, we obtain results as follow (take ChatGPT as an example)
|Method|$A_{i.d.}^{original}(\uparrow)$|$P_{i.d.}^{original}(\downarrow)$|$A_{i.d.}^{holdout}(\uparrow)$|$P_{i.d.}^{holdout}(\downarrow)$|$A_{comp}^{holdout}(\uparrow)$|$P_{comp}^{holdout}(\downarrow)$|$A_{i.d.}^{acd}(\uparrow)$|$P_{i.d.}^{acd}(\downarrow)$|$A_{comp}^{acd}(\uparrow)$|$P_{comp}^{acd}(\downarrow)$|$A_{avg}(\uparrow)$|$P_{avg}(\downarrow)$|$G_{avg}(\downarrow)$|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|_ChatGPT_|57.51%|18.03|56.62%|18.29|49.21%|18.49|57.13%|18.27|49.75%|18.22|54.04%|18.26|13.00%|

- $A_{avg}=\frac{1}{5}(A_{i.d.}^{original}+A_{i.d.}^{holdout}+A_{comp}^{holdout}+A_{i.d.}^{acd}+A_{comp}^{acd})$
- $P_{avg}=\frac{1}{5}(P_{i.d.}^{original}+P_{i.d.}^{holdout}+P_{comp}^{holdout}+P_{i.d.}^{acd}+P_{comp}^{acd})$
- $G_{avg}=\frac{1}{2}(\frac{A_{i.d.}^{holdout}-A_{comp}^{holdout}}{A_{i.d.}^{holdout}}+\frac{A_{i.d.}^{acd}-A_{comp}^{acd}}{A_{i.d.}^{acd}})$

where each accuracy and perplexity in the table above (except for $A_{avg}$ and $P_{avg}$) are obtained by averaging the accuracies across four datasets (Fyelp, Amazon, YELP, Mixture):
- $A_{type}^{mode}=\frac{1}{4}(A_{type}^{mode}(Fyelp)+A_{type}^{mode}(Amazon)+A_{type}^{mode}(YELP)+A_{type}^{mode}(Mixture))$
- $P_{type}^{mode}=\frac{1}{4}(P_{type}^{mode}(Fyelp)+P_{type}^{mode}(Amazon)+P_{type}^{mode}(YELP)+P_{type}^{mode}(Mixture))$

where $type\in\{i.d., comp\},\quad mode\in\{original, holdout, acd\},\quad (comp, original)\notin(type, mode)$
