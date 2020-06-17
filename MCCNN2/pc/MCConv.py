class MCConv:

    def __init__(self, pNumInFeatures, pNumOutFeatures, pHiddenSize, pNumDims, pConvName):

        self.numInFeatures_ = pNumInFeatures
        self.numOutFeatures_ = pNumOutFeatures
        self.numHidden_ = pHiddenSize
        self.numDims_ = pNumDims
        self.convName_ = pConvName

        stdDev = math.sqrt(1.0/float(self.numDims_))
        hProjVecTF = tf.get_variable(self.convName_+'_hidden_vectors', shape=[self.numHidden_, self.numDims_], 
            initializer=tf.initializers.truncated_normal(stddev=stdDev), dtype=tf.float32, trainable=True)
        hProjBiasTF = tf.get_variable(self.convName_+'_hidden_biases', shape=[self.numHidden_, 1], 
            initializer=tf.initializers.zeros(), dtype=tf.float32, trainable=True)
        self.basisTF_ = tf.concat([hProjVecTF, hProjBiasTF], axis = 1)

        stdDev = math.sqrt(2.0/float(self.numHidden_*self.numInFeatures_))
        self.weights_ = tf.get_variable(self.convName_+'_conv_weights',
            shape=[self.numHidden_ * self.numInFeatures_, self.numOutFeatures_], 
            initializer=tf.initializers.truncated_normal(stddev=stdDev), 
            dtype=tf.float32, trainable=True)


    def __call__(self, 
        pInFeatures,
        pInPC,
        pOutPC,
        pRadius):

        #Create the radii tensor.
        curRadii = np.array([pRadius for i in range(self.numDims_)])
        radiiTensor = tf.convert_to_tensor(curRadii, np.float32)

        #Create the badnwidth tensor.
        curBandWidth = np.concatenate([0.2 for i in range(self.numDims_)])
        bwTensor = tf.convert_to_tensor(curBandWidth, np.float32)

        #Compute the AABB.
        aabbIn = AABB(pInPC)

        #Compute the grid.
        grid = Grid(pInPC, aabbIn, radiiTensor)

        #Compute the neighborhood key.
        neigh = Neighborhood(grid, radiiTensor, pOutPC, 0)
        neigh.compute_pdf(bwTensor, pKDEMode, True)

        #Compute convolution (RELU - 2, LRELU - 3, ELU - 4)
        inWeightFeat = basis_proj(neigh, pInFeatures, self.basisTF_, 3)
            
        #Compute the convolution.
        return tf.matmul(tf.reshape(inWeightFeat, [-1, self.numInFeatures_*self.numHidden_]), self.weights_)