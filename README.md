<!--
MIT License

Copyright (c) 2018 Othneil Drew

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
-->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

<p align="center">
  <a href="https://github.com/marakeby/pnet_prostate_paper">
    <img src="_plots/screenshot.png" alt="Logo" width="900" height="300">
  </a>
  </p>


Biologically informed deep learning for explainable epigenetic clocks

<!-- GETTING STARTED -->

## Getting Started




### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Paureel/XAI-AGE.git
   ```
2. Create conda environment
   ```sh
   conda env create --name age_env --file=environment.yml
   ```


<!-- USAGE EXAMPLES -->

## Usage

1. Activate the created conda environment
   ```sh
   source activate age_env
   ```
2. Add the current diretory to PYTHONPATH, e.g.

   ```sh
   export PYTHONPATH=~/pnet_prostate_paper:$PYTHONPATH
   ```

3. To generate all paper figures, run
     ```sh
   cd ./analysis
   python run_it_all.py
   ```

4. To generate individual paper figure run the different files under the 'analysis' directory, e.g.
     ```sh
   cd ./analysis
   python figure_1_d_auc_prc.py
   ```
   For ```Figure3``` , make sure you run ```prepare_data.py``` before running other files
5. To re-train a model from scratch run
   ```sh
   cd ./train
   python run_me.py
   ```
   This will run an experiment 'pnet/onsplit_average_reg_10_tanh_large_testing' which trains a P-NET model on a
   training-testing data split of Armenia et al data set and compare it to a simple logistic regression model. The
   results of the experiment will be stored under ```_logs```in a directory with the same name as the experiment.  
   To run another experiment, you may uncomment one of the lines in the run_me.py to run the corresponding experiment.
   Note that some models especially cross validation experiments may be time consuming.

<!-- LICENSE -->

## License

Distributed under the GPL-2.0 License License. See `LICENSE` for more information.



<!-- CONTACT -->

## Contact

Haitham - [@HMarakeby](https://twitter.com/HMarakeby)

Project Link: [https://github.com/marakeby/pnet_prostate_paper](https://github.com/marakeby/pnet_prostate_paper)


<!-- References -->

## References
* Elmarakeby H, et al. "Biologically informed deep neural network for prostate cancer classification and discovery." Nature. Online September 22, 2021. DOI: 10.1038/s41586-021-03922-4
* Armenia, Joshua, et al. "The long tail of oncogenic drivers in prostate cancer." Nature genetics 50.5 (2018): 645-651.
* Robinson, Dan R., et al. "Integrative clinical genomics of metastatic cancer." Nature 548.7667 (2017): 297-303.
* Fraser, Michael, et al. "Genomic hallmarks of localized, non-indolent prostate cancer." Nature 541.7637 (2017):
  359-364.

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements
This work was supported in part by the Fund for Innovation in Cancer Informatics, Mark Foundation, Prostate Cancer Foundation, Movember, and the National Cancer Institute at the National Institutes of Health.
