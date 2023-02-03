<h1>Neural Logic Circuits (NLC)</h1>
<p>Hamit Taner Ünal & Prof.Fatih Başçiftçi</p>
<p></p>
Neural Logic Circuits (NLC) is an evolutionary, weightless, and learnable neural architecture loosely inspired by the neuroplasticity of the brain. This new paradigm achieves learning by evolution of its architecture through reorganization of augmenting synaptic connections and generation of artificial neurons functioning as logic gates. These neural units mimic biological nerve cells which are stimulated by binary input signals and emit excitatory or inhibitory pulses, thus executing the “all-or-none” character of their natural counterparts. Unlike Artificial Neural Networks (ANN), our model achieves generalization ability without intensive weight training, and dedicates computational resources solely to building network architecture with optimal connectivity.

<h2>Variable-Length Direct Encoding</h2>
Here we adopt a direct encoding approach similar to the pioneering work of Miller et al. (1989). In this method, each network design is represented by a connectivity matrix that is mapped directly into a genotype, built with an array of bit strings. We propose variable-length chromosome encoding which determines the size of the matrix depending on the number of gates used. Each entry on the connectivity matrix indicates the existence of neural connectivity between two units (true (1) if a connection exists and false (0) for no connection). Then, successive columns are concatenated to build the genotype. 
<p></p>
Unlike conventional evolutionary approaches, we used two separate chromosomes (Chromosome A and Chromosome B) in synchronization with each other for every individual in the population. The first chromosome is used for gate types (neurons). The latter represents gate connections (synapses). In NLC, A logic gate simulates the behavior of biological neurons and can have six different types, which are AND, OR, NAND, NOR, XOR, and XNOR. In our implementation, each gate is represented with an integer number between 0 and 5.  

<img width="306" alt="image" src="https://user-images.githubusercontent.com/30185318/216526553-af3c8ec1-b03f-4d88-890a-1cff00a00cda.png">
<p></p>
The second chromosome (B), which represents the connectivity of neurons contains only binary values, either true (1) or false (0). The evolutionary mechanism starts with a pre-determined number of gates and the network is gradually augmented, starting from a minimal structure. A column is generated for each gate added to the network. Each row in this column has an entry for previous neural units, as well as sensory inputs. The first gate  in the network has only rows for input units. The second gate   has a row for  (to indicate connectivity), in addition to inputs. The last gate in the network is the output gate. Every gate, as well as the output gate, can have connections from all previous neural units, including the inputs. So, a direct connection from input to any unit is possible. A sample network with two inputs, five gates, and single output is explained below.

<img width="643" alt="image" src="https://user-images.githubusercontent.com/30185318/216526740-fc882b1b-ff13-4b2e-9274-c533a37f7f26.png">
<p></p>
Unlike Artificial Neural Networks, an NLC network doesn’t have layers. Neural units are distributed spatially in three dimensions and there is no constraint in connectivity. Gates are placed in hierarchical order and represented in the chromosome in the order they added to the network. The combinational circuit of the sample NLC network given above is illustrated below.

<img width="731" alt="image" src="https://user-images.githubusercontent.com/30185318/216526891-c1733231-5608-47eb-a43d-0e85b48a9b9b.png">

The output of an NLC network is calculated in a forward flow, starting from the sensory inputs and taking outputs of each gate hierarchically. Unlike Artificial Neural Networks, an NLC network doesn’t have layers and there is no backward flow. Similar to biological neurons, NLC neural units are distributed spatially in three dimensions and there is no constraint on connectivity. As shown in figure below, the output  of each gate   becomes available for the following gates if a connection exists.

<img width="815" alt="image" src="https://user-images.githubusercontent.com/30185318/216527040-286886fa-6d37-4e11-a291-3daaf0bc0f2d.png">

<p></p>
<h1>Converting Datasets to Binary</h1>

The binary system is the internal language of NLC. Prior to conducting experiments, we converted all data in the datasets to binary. For this purpose, we analyzed every input variable by grouping them into several categories. First of all, we used one-hot-encoding to convert categorical data to binary. Then, we converted integer and real-valued data into binary directly and determined the length of the input column from the maximum value of each variable. If the input contains negative values, we added a sign bit. It is important to keep the length of each column fixed and we aimed to minimize input dimensions while keeping the network sparse. Here is an illustration, how we performed this step with examples:
<p></p>
- Consider converting ‘Pregnancies’ attribute to binary in Pima Indian Diabetes dataset. The maximum value is 17. When we convert 17 to binary we obtain ‘10001’. The length is 5 bits. Then, we convert each input in 5 digits. For instance; if the attribute has a value of 3, the input column takes the value of ‘00011’ (not ‘11’). 
<p></p>
- For floating numbers, first we convert the value to plain decimals. For example, consider converting ‘BMI’ attribute to binary in Pima Indian Diabetes. Maximum value for this attribute is 67.1. Here the precision is 0.1 and we multiply the values with 10 to obtain plain decimal numbers. In this case, we multiply 67.1 with 10 and get 671. We convert 671 to binary and get ‘1010011111’. The outcome has 10 digits. If we encounter a negative value in the attribute, we add a sign bit and input column becomes 11 bit (it was not the case for BMI). Similar to previous explanation, we convert each entry in the dataset to binary and make it 10 bits by adding zeros in front of it.   
<p></p>

<h2>Instructions for C++ Files</h2>
All you need to do is to copy and paste the cpp files into a text editor or C++ IDE, such as CLion. Make sure you keep the dataset CSV file at the same folder.




