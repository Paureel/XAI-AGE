


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
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



**Biologically informed deep learning for explainable epigenetic clocks**

Aging is defined by steady buildup of damage and is a risk factor for chronic diseases. Epigenetic mechanisms like DNA methylation may play a role in organismal aging, but whether they are active drivers or consequences is unknown. Epigenetic clocks, based on DNA methylation, accurately determine a person's biological age. In the past years, a number of accurate epigenetic clocks were developed, and their function and an overview of the field is summarized by Seale et al.. 
         
Here we present **XAI-AGE**, which is a biologically informed, explainable deep neural network model for accurate biological age prediction across many tissues. We show that this approach can identify differentially activated pathways and biological processes from the latent layers of the neural network, and is based on a recently published explainable model used in cancer research, called PNET by Elmarakeby et al.. 

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
   conda install -c anaconda ipykernel
   python -m ipykernel install --user --name=age_env
   ```
   
3. Download the remaining files from: 
Dropbox link: https://www.dropbox.com/scl/fi/ni9frchnalyw9c9fj7xco/_database.zip?rlkey=0nngaa2uw14vff23uuembgdjb&dl=0


<!-- USAGE EXAMPLES -->

## Usage

1. Activate the created conda environment
   ```sh
   source activate age_env
   ```

2. Follow the instructions in the make_individual_predictions.ipynb file.

3. Generate the Sankey diagram with the generate_sankey.ipynb file.

<!-- LICENSE -->

## License

Distributed under the GPL-2.0 License. The changed files compared to the original PNET publication was marked in every affected files.



<!-- CONTACT -->

## Contact


Project Link: [https://github.com/Paureel/XAI-AGE](https://github.com/Paureel/XAI-AGE)


<!-- References -->

## References
* Elmarakeby H, et al. "Biologically informed deep neural network for prostate cancer classification and discovery." Nature. Online September 22, 2021. DOI: 10.1038/s41586-021-03922-4
* Seale et al. "Making sense of the ageing methylome",Nature Reviews Genetics, DOI:10.1038/s41576-022-00477-6


<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements
Funded by the MILAB Artificial Intelligence National Laboratory Program of the Ministry of Innovation and Technology from the National Research, Development and Innovation Fund.