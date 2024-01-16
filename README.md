# TopoLayer
![TopoLayer](https://github.com/Anonymous-ijcai-1/TopoLayer/blob/main/images/Topolayer.jpg)

The illustration depicts the pipeline for extracting topological features within a point cloud and the neural network architecture of TopoLayer. This pipeline consists of two main steps: (1) The generation of persistence diagrams derived from the filtration of Vietoris-Rips complexes applied to a point cloud; (2) The generation of topology features by TopoLayer. For various dimensional persistence diagrams, TopoLayer can employ PPDTF, PDTF, or a combination of both methods to vectorize these persistence diagrams. Consequently, a weight function denoted as $\omega\left(\cdot\right)$ is applied to each resulting vector, and a permutation-invariant operation represented as \textbf{op} is utilized to extract topology features. Subsequently, these topology features are concatenated to obtain a global topology feature. This process offers a range of parameter options, providing flexibility and adaptability.

## Result
Here are files of the experiment results: [Result](https://drive.google.com/drive/folders/1iFS2vJjwxr5lBL0OXoVkO3KaFpZ04v_n?usp=sharing)
