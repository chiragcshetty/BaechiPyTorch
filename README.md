# Baechi-PyTorch: Automated Model Parallelism in PyTorch

<span style="color:blue">What?</span>. To train large DNN's over GPUs with limited memory, the model must be split across multiple devices - Model Parallelism. Similarly, training times can be reduced by distributing parallel branches on the model across the devices. 

Currently, the process is manual and largely based on heuristics, as we demonstrate [here](https://github.com/chiragcshetty/BaechiPyTorch/blob/669d3d241a9b95dea957c4ccc2ec585ec7ccb15e/docs/Baechi_pytorch_system_design.pdf) (Section 1.2) 


In Baechi, we adopt an algorithmic approach to the placement problem for running DNN training graphs on a small cluster of memory-constrained devices. **Baechi-PyTorch , automatically and optimally splits the model, given a number of GPU devices and their memory capacities.** 

Please find the design and usage information for Baechi-PyTorch here: [link](https://scientific-goldfish-3af.notion.site/Baechi-PyTorch-8703ed020ce04f83b956231743b4e898)


Tensorflow implementation of Baechi can be found here: [Baechi](https://dprg.cs.uiuc.edu/downloads.php) <br />
The corresponding [paper](https://dl.acm.org/doi/10.1145/3419111.3421302) presented at SoCC 2020. <br />

Draft of Baechi Extended version paper is [here](https://www.chiragshetty.com/pdf/baechi_extended.pdf). (Currently, under review) <br />

For any queries, suggestions etc please feel free to reach out at cshetty2@illinois.edu 
