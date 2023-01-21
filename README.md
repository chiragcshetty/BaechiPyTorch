# BaechiPyTorch


Deep Neural Networks can be challenging or impossible to train when either devices have limited memory, or the models are large. Splitting the model graph across multiple devices, today, is largely heuristic based and manual. We present the Baechi system, where we adopt an algorithmic approach to the placement problem for running machine learning training graphs on a small cluster of memory-constrained devices. **Baechi-PyTorch , automatically and optimally splits the model, given a number of GPU devices and their memory capacities.** 

Please find the design and usage information for Baechi-PyTorch here: [link](https://scientific-goldfish-3af.notion.site/Baechi-PyTorch-8703ed020ce04f83b956231743b4e898)


Tensorflow implementation of Baechi can be found here: [Baechi](https://dprg.cs.uiuc.edu/downloads.php) <br />
The corresponding [paper](https://dl.acm.org/doi/10.1145/3419111.3421302) presented at SoCC 2020. <br />

Draft of Baechi Extended version paper is [here](https://www.chiragshetty.com/pdf/baechi_extended.pdf). (Currently, under review) <br />

For any queries, suggestions etc please feel free to reach out at cshetty2@illinois.edu 
