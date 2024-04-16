# CompMCTG Benchmark \& Meta-MCTG
## 1. How to use our CompMCTG Benchmark?
For each method, we obtain results as follow (take ChatGPT as an example)
|Method|$A_{i.d.}^{Original}(\uparrow)$|$P_{i.d.}^{Original}(\downarrow)$|$A_{i.d.}^{Holdout}(\uparrow)$|$P_{i.d.}^{Holdout}(\downarrow)$|$A_{comp}^{Holdout}(\uparrow)$|$P_{comp}^{Holdout}(\downarrow)$|$A_{i.d.}^{ACD}(\uparrow)$|$P_{i.d.}^{ACD}(\downarrow)$|$A_{comp}^{ACD}(\uparrow)$|$P_{comp}^{ACD}(\downarrow)$|$A_{avg}(\uparrow)$|$P_{avg}(\downarrow)$|$G_{avg}(\downarrow)$|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|_ChatGPT_|57.51%|18.03|56.62%|18.29|49.21%|18.49|57.13%|18.27|49.75%|18.22|54.04%|18.26|13.00|

- $A_{avg}=\frac{1}{5}(A_{i.d.}^{Original}+A_{i.d.}^{Holdout}+A_{comp}^{Holdout}+A_{i.d.}^{ACD}+A_{comp}^{ACD})$<\br>
- $P_{avg}=\frac{1}{5}(P_{i.d.}^{Original}+P_{i.d.}^{Holdout}+P_{comp}^{Holdout}+P_{i.d.}^{ACD}+P_{comp}^{ACD})$<\br>
