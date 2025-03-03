package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"

	"phase" // Adjust import to your local path
)

// ============================================================================
// Constants & Types
// ============================================================================

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"

	// Evolution hyperparameters
	populationSize      = 50
	numGenerations      = 3000
	selectionPercentage = 0.3
	topPercentage       = 0.05 // Top 5% of models to attempt further improvement
	mutationRate        = 0.4  // Probability of structural changes per child

	// We do not do gradient-based backprop – purely random neuroevolution
	// If you decide to do partial training, you can insert it here.

	// For saving state
	saveFolder  = "test"
	stateFile   = "nas_population.json"
	resultsFile = "results.json"

	stagnationCounterMax = 5

	// Complexity-based neuron addition range
	Complexity1MinNeurons = 1
	Complexity1MaxNeurons = 6
	Complexity2MinNeurons = 2
	Complexity2MaxNeurons = 10
	Complexity3MinNeurons = 3
	Complexity3MaxNeurons = 15

	// For “rush mode” – focusing on proximity when exact < 60%
	rushModeEnabled = true
	rushThreshold   = 0.6

	// Stop early if we surpass some high threshold
	stopEarlyExactAcc = 0.75

	// For partial test subset
	testSubsetSize = 3000 // Evaluate on 3k from the test set

	// For connection settings (when adding new neurons)
	minConnections = 5
	maxConnections = 40

	maxClamp = 1e4
	minClamp = -1e4

	closeThreshold = 0.9

	// For targeted mutation
	maxAttempts            = 3
	enableTargetedMutation = false // Turn on or off for that logic

	// Adjustment sweep for top models
	adjustmentMin  = -0.05
	adjustmentMax  = 0.05
	adjustmentStep = 0.01
)

// EvolutionaryState is what we save to JSON
type EvolutionaryState struct {
	Population        []*phase.Phase
	Generation        int
	BestFitness       float64
	ComplexityLevel   int
	StagnationCounter int
	KnownSamples      map[int]bool
	BestModel         *phase.Phase
}

// Results holds final stats
type Results struct {
	TrainExactAcc       float64
	TrainCloseAcc       float64
	TrainProximityScore float64
	TestExactAcc        float64
	TestCloseAcc        float64
	TestProximityScore  float64
}

// BestModelTracker for concurrency
type BestModelTracker struct {
	mu                 sync.Mutex
	overallBestModel   *phase.Phase
	bestExactAcc       float64
	bestCloseAcc       float64
	bestProximityScore float64
	updated            bool
}

type AdjustmentTask struct {
	adjustWeights bool
	adjustBiases  bool
	adjValue      float64
}

type AdjustmentResult struct {
	model          *phase.Phase
	fitness        float64
	proximityScore float64
}

// ============================================================================
// Main
// ============================================================================

func main() {
	prev := runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Printf("Previous GOMAXPROCS: %d, now set to %d\n", prev, runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())

	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Load full training + testing data
	trainXAll, trainYAll, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	testXAll, testYAll, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading testing MNIST: %v", err)
	}

	// Combine + shuffle, then do 80-20 for train/test
	allX := mat.NewDense(70000, 784, nil)
	allX.Stack(trainXAll, testXAll)
	allY := mat.NewDense(70000, 1, nil)
	allY.Stack(trainYAll, testYAll)

	perm := rand.Perm(70000)
	shuffledX := mat.NewDense(70000, 784, nil)
	shuffledY := mat.NewDense(70000, 1, nil)
	for i, p := range perm {
		shuffledX.SetRow(i, allX.RawRowView(p))
		shuffledY.Set(i, 0, allY.At(p, 0))
	}

	trainSamples := int(0.8 * 70000.0)
	testSamples := 70000 - trainSamples

	trainX := mat.NewDense(trainSamples, 784, nil)
	trainY := mat.NewDense(trainSamples, 1, nil)
	testX := mat.NewDense(testSamples, 784, nil)
	testY := mat.NewDense(testSamples, 1, nil)

	trainX.Copy(shuffledX.Slice(0, trainSamples, 0, 784))
	trainY.Copy(shuffledY.Slice(0, trainSamples, 0, 1))
	testX.Copy(shuffledX.Slice(trainSamples, 70000, 0, 784))
	testY.Copy(shuffledY.Slice(trainSamples, 70000, 0, 1))

	if err := os.MkdirAll(saveFolder, 0755); err != nil {
		log.Fatalf("Failed to create folder %s: %v", saveFolder, err)
	}

	trainTesting(trainX, trainY, testX, testY)
	fmt.Println("Done.")
}

// ============================================================================
// Step 1: Data + Download
// ============================================================================

func ensureMNISTDownloads(bp *phase.Phase, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}
	for _, f := range files {
		localPath := filepath.Join(targetDir, f)
		if _, err := os.Stat(localPath); os.IsNotExist(err) {
			fmt.Printf("Downloading %s...\n", f)
			if err := bp.DownloadFile(localPath, baseURL+f); err != nil {
				return fmt.Errorf("failed to download %s: %w", f, err)
			}
			fmt.Printf("Unzipping %s...\n", f)
			if err := bp.UnzipFile(localPath, targetDir); err != nil {
				return fmt.Errorf("failed to unzip %s: %w", f, err)
			}
		}
	}
	return nil
}

func loadMNIST(dir, imageFile, labelFile string, limit int) (*mat.Dense, *mat.Dense, error) {
	imgPath := filepath.Join(dir, imageFile)
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open image file: %w", err)
	}
	defer fImg.Close()

	var headerImg [16]byte
	if _, err := fImg.Read(headerImg[:]); err != nil {
		return nil, nil, fmt.Errorf("read image header: %w", err)
	}
	numImages := int(binary.BigEndian.Uint32(headerImg[4:8]))
	numRows := int(binary.BigEndian.Uint32(headerImg[8:12]))
	numCols := int(binary.BigEndian.Uint32(headerImg[12:16]))
	if numRows != 28 || numCols != 28 {
		return nil, nil, fmt.Errorf("expected 28x28, got %dx%d", numRows, numCols)
	}
	if limit > numImages {
		limit = numImages
	}

	lblPath := filepath.Join(dir, labelFile)
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open label file: %w", err)
	}
	defer fLbl.Close()

	var headerLbl [8]byte
	if _, err := fLbl.Read(headerLbl[:]); err != nil {
		return nil, nil, fmt.Errorf("read label header: %w", err)
	}
	numLabels := int(binary.BigEndian.Uint32(headerLbl[4:8]))
	if limit > numLabels {
		limit = numLabels
	}

	dataX := mat.NewDense(limit, 784, nil)
	dataY := mat.NewDense(limit, 1, nil)

	buf := make([]byte, 784)
	for i := 0; i < limit; i++ {
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data: %w", err)
		}
		for j := 0; j < 784; j++ {
			dataX.Set(i, j, float64(buf[j])/255.0)
		}
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data: %w", err)
		}
		dataY.Set(i, 0, float64(lblByte[0]))
	}
	return dataX, dataY, nil
}

// ============================================================================
// Step 2: The Evolutionary “Main Loop”
// ============================================================================

func trainTesting(trainX, trainY, testX, testY *mat.Dense) {
	fmt.Println("Starting evolutionary run:", time.Now())

	cpuCount := runtime.NumCPU()
	targetCores := int(math.Floor(0.8 * float64(cpuCount)))
	if targetCores < 1 {
		targetCores = 1
	}
	fmt.Printf("Using %d cores (80%% of %d CPU cores)\n", targetCores, cpuCount)

	var population []*phase.Phase
	var gen int
	var bestFitness float64
	var complexityLevel int
	var stagnationCounter int
	knownSamples := make(map[int]bool)
	var bestModel *phase.Phase

	// Attempt to load a saved state
	statePath := filepath.Join(saveFolder, stateFile)
	if _, err := os.Stat(statePath); err == nil {
		st, err := loadState(stateFile)
		if err != nil {
			log.Fatalf("Failed to load state: %v", err)
		}
		population = st.Population
		gen = st.Generation
		bestFitness = st.BestFitness
		complexityLevel = st.ComplexityLevel
		stagnationCounter = st.StagnationCounter
		knownSamples = st.KnownSamples
		bestModel = st.BestModel
		if knownSamples == nil {
			knownSamples = make(map[int]bool)
		}
		if bestModel == nil && len(population) > 0 {
			bestModel = population[0].Copy()
		}
		fmt.Printf("Resuming from generation %d with best fitness %.4f\n", gen, bestFitness)
	} else {
		population = createInitialPopulation(populationSize)
		gen = 0
		bestFitness = 0.0
		complexityLevel = 1
		stagnationCounter = 0
		bestModel = population[0].Copy()
	}

	bestModelTracker := &BestModelTracker{
		bestExactAcc:       -1,
		bestCloseAcc:       -1,
		bestProximityScore: -1,
		updated:            false,
		overallBestModel:   bestModel,
	}

	for gen < numGenerations {
		fmt.Printf("\n=== Generation %d (Complexity: %d) ===\n", gen, complexityLevel)
		startGenTime := time.Now()

		// Evaluate population
		fitness, exactAccs, closeAccs, proxScores := evaluatePopulation(population, testX, testY, targetCores, bestModelTracker)

		// Show stats
		currentBest, bestIndex := maxFitness(fitness)
		avgFit, minFit := averageMinFitness(fitness)
		fmt.Printf("   => Best Fitness: %.4f  [Exact=%.2f%%, Close=%.2f%%, Prox=%.2f%%]\n",
			currentBest, exactAccs[bestIndex]*100, closeAccs[bestIndex]*100, proxScores[bestIndex])
		fmt.Printf("   => Avg Fitness:  %.4f  | Min Fitness: %.4f\n", avgFit, minFit)

		if err := saveModel(population[bestIndex], gen, currentBest, currentBest > bestFitness); err != nil {
			log.Printf("Failed to save model (gen %d): %v", gen, err)
		}

		// Possibly early-stop if we have high exact accuracy
		if bestModelTracker.bestExactAcc >= stopEarlyExactAcc {
			bestFitness = currentBest
			if bestModelTracker.overallBestModel == nil {
				bestModelTracker.overallBestModel = population[bestIndex].Copy()
			}
			fmt.Printf("Reached %.2f%% exact accuracy; stopping early at generation %d.\n",
				bestModelTracker.bestExactAcc*100, gen)
			finishAndExit(population, gen+1, bestFitness, complexityLevel, stagnationCounter,
				bestModelTracker.overallBestModel, trainX, trainY, testX, testY, knownSamples)
			return
		}

		// Update known samples in case we do targeted mutation
		updateKnownSamples(bestModelTracker.overallBestModel, testX, testY, knownSamples)

		// If we found a new best this generation:
		if currentBest > bestFitness {
			bestFitness = currentBest
			stagnationCounter = 0
		} else {
			stagnationCounter++
		}

		fmt.Printf("   => Stagnation: %d/%d\n", stagnationCounter, stagnationCounterMax)
		if stagnationCounter >= stagnationCounterMax {
			complexityLevel++
			if complexityLevel > 5 {
				complexityLevel = 5 // cap it
			}
			stagnationCounter = 0
			fmt.Printf("   => Complexity raised to %d\n", complexityLevel)
		}

		// Improve top fraction with small weight/bias adjustments
		topModels := selectTopPerformers(population, exactAccs, closeAccs, proxScores, topPercentage)
		isRushMode := rushModeEnabled && bestModelTracker.bestExactAcc < rushThreshold

		// Decide on ranges for adjacency
		var adjMin, adjMax, adjStep float64
		if isRushMode {
			adjMin = -0.1
			adjMax = 0.1
			adjStep = 0.02
		} else {
			adjMin = adjustmentMin
			adjMax = adjustmentMax
			adjStep = adjustmentStep
		}
		adjustments := generateAdjustments(adjMin, adjMax, adjStep)
		for i, m := range topModels {
			bestAdjModel, bestAdjFit, bestAdjProx := findBestAdjustment(m, adjustments, testX, testY, isRushMode)
			origFit := fitness[i]
			origProx := proxScores[i]
			if isRushMode {
				if bestAdjProx > origProx {
					topModels[i] = bestAdjModel
					fmt.Printf("   => Rush improvement: proximity %.2f -> %.2f\n", origProx, bestAdjProx)
				}
			} else {
				if bestAdjFit > origFit {
					topModels[i] = bestAdjModel
					fmt.Printf("   => Swept improvement: fit %.4f -> %.4f\n", origFit, bestAdjFit)
				}
			}
		}

		// Next generation
		selected := selectTopPerformers(population, exactAccs, closeAccs, proxScores, selectionPercentage)
		population = createNextGeneration(selected, populationSize, mutationRate, complexityLevel, bestModelTracker.overallBestModel, knownSamples, testX, testY)

		// Overwrite top portion with improved top models
		for i := 0; i < len(topModels) && i < len(population); i++ {
			population[i] = topModels[i]
		}

		// Save state
		st := EvolutionaryState{
			Population:        population,
			Generation:        gen + 1,
			BestFitness:       bestFitness,
			ComplexityLevel:   complexityLevel,
			StagnationCounter: stagnationCounter,
			KnownSamples:      knownSamples,
			BestModel:         bestModelTracker.overallBestModel,
		}
		if err := saveState(st, stateFile); err != nil {
			log.Printf("Failed to save state: %v", err)
		}

		elapsed := time.Since(startGenTime)
		fmt.Printf("Generation %d completed in %v\n", gen, elapsed)
		gen++
	}

	// Done with all gens
	fmt.Println("Max generations reached.")
	trainExact, trainClose, trainProx := evaluateAccuracy(bestModelTracker.overallBestModel, trainX, trainY)
	testExact, testClose, testProx := evaluateAccuracy(bestModelTracker.overallBestModel, testX, testY)
	results := Results{
		TrainExactAcc:       trainExact,
		TrainCloseAcc:       trainClose,
		TrainProximityScore: trainProx,
		TestExactAcc:        testExact,
		TestCloseAcc:        testClose,
		TestProximityScore:  testProx,
	}
	if err := saveResults(results, resultsFile); err != nil {
		log.Printf("Failed to save results: %v", err)
	}
	visualizeResults(bestModelTracker.overallBestModel, trainX, trainY, testX, testY)
	fmt.Printf("Final best fitness after %d generations: %.4f\n", gen, bestFitness)
}

// ============================================================================
// Step 3: The Generation Tools – Evaluate, Select, Mutate, Crossover
// ============================================================================

// Evaluate entire population concurrency
func evaluatePopulation(pop []*phase.Phase, testX, testY *mat.Dense, cores int, bestModelTracker *BestModelTracker) ([]float64, []float64, []float64, []float64) {
	nPop := len(pop)
	fitness := make([]float64, nPop)
	exactArr := make([]float64, nPop)
	closeArr := make([]float64, nPop)
	proxArr := make([]float64, nPop)

	semaphore := make(chan struct{}, cores)
	var wg sync.WaitGroup

	// Use a random subset of the test set for speed if the test set is big
	subX, subY := sampleTestSet(testX, testY, testSubsetSize)

	for i := 0; i < nPop; i++ {
		wg.Add(1)
		semaphore <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-semaphore }()
			bp := pop[idx]
			exactAcc, closeAcc, proxScore := evaluateAccuracy(bp, subX, subY)
			exactArr[idx] = exactAcc
			closeArr[idx] = closeAcc
			proxArr[idx] = proxScore
			fitness[idx] = 0.5*exactAcc + 0.3*closeAcc + 0.2*(proxScore/100.0)

			// Possibly update best overall
			bestModelTracker.mu.Lock()
			reason := ""
			if rushModeEnabled && bestModelTracker.bestExactAcc < rushThreshold {
				// focus on proximity
				if proxScore > bestModelTracker.bestProximityScore {
					reason = "proximity improved (rush mode)"
				}
			} else {
				// prefer exact
				if exactAcc > bestModelTracker.bestExactAcc {
					reason = "exact improved"
				} else if exactAcc == bestModelTracker.bestExactAcc && closeAcc > bestModelTracker.bestCloseAcc {
					reason = "close improved"
				} else if exactAcc == bestModelTracker.bestExactAcc && closeAcc == bestModelTracker.bestCloseAcc && proxScore > bestModelTracker.bestProximityScore {
					reason = "proximity improved"
				}
			}
			if reason != "" {
				bestModelTracker.overallBestModel = bp.Copy()
				bestModelTracker.bestExactAcc = exactAcc
				bestModelTracker.bestCloseAcc = closeAcc
				bestModelTracker.bestProximityScore = proxScore
				bestModelTracker.updated = true
				fmt.Printf("New Best Model (%s): exact=%.2f%%, close=%.2f%%, prox=%.2f%%\n",
					reason, exactAcc*100, closeAcc*100, proxScore)
			}
			bestModelTracker.mu.Unlock()

		}(i)
	}
	wg.Wait()
	return fitness, exactArr, closeArr, proxArr
}

// sampleTestSet picks a random subset of the test set
func sampleTestSet(X, Y *mat.Dense, subset int) (*mat.Dense, *mat.Dense) {
	r, _ := X.Dims()
	if subset >= r {
		return X, Y
	}
	perm := rand.Perm(r)
	subX := mat.NewDense(subset, 784, nil)
	subY := mat.NewDense(subset, 1, nil)
	for i := 0; i < subset; i++ {
		srcRow := perm[i]
		subX.SetRow(i, X.RawRowView(srcRow))
		subY.Set(i, 0, Y.At(srcRow, 0))
	}
	return subX, subY
}

// createInitialPopulation with a baseline architecture
func createInitialPopulation(popSize int) []*phase.Phase {
	out := make([]*phase.Phase, popSize)
	for i := 0; i < popSize; i++ {
		bp := phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
		out[i] = bp
	}
	return out
}

// createNextGeneration – uses both mutation & optional crossover
func createNextGeneration(parents []*phase.Phase, popSize int, mutationRate float64, cLevel int,
	overallBest *phase.Phase, knownSamples map[int]bool, testX, testY *mat.Dense) []*phase.Phase {

	nextPop := make([]*phase.Phase, popSize)
	// Keep overall best in slot 0
	nextPop[0] = overallBest.Copy()

	// In the rest, do a blend of reproduction + crossover
	for i := 1; i < popSize; i++ {
		parentA := parents[rand.Intn(len(parents))]
		child := parentA.Copy()

		if rand.Float64() < 0.5 && len(parents) > 1 {
			// Attempt crossover
			parentB := parents[rand.Intn(len(parents))]
			if parentA != parentB {
				child = crossoverPhases(parentA, parentB)
			}
		}

		// Probability of mutation
		if rand.Float64() < mutationRate {
			structuralMutate(child, cLevel)
		}
		pruneLowWeightNeurons(child)

		nextPop[i] = child
	}
	return nextPop
}

// structuralMutate picks random structural changes
func structuralMutate(bp *phase.Phase, complexity int) {
	roll := rand.Intn(5)
	switch roll {
	case 0:
		// Add a few neurons
		nMin, nMax := getNeuronRangeByComplexity(complexity)
		toAdd := rand.Intn(nMax-nMin+1) + nMin
		fmt.Printf("   => Adding %d random neurons\n", toAdd)
		for i := 0; i < toAdd; i++ {
			bp.AddRandomNeuron("", "", minConnections, maxConnections)
			bp.RewireOutputsThroughNewNeuron(bp.GetNextNeuronID() - 1)
		}
	case 1:
		// Add a new connection
		bp.AddConnection()
	case 2:
		// Remove a random connection
		bp.RemoveConnection()
	case 3:
		// Slightly adjust all weights
		delta := rand.NormFloat64() * 0.02
		bp.AdjustAllWeights(delta)
	case 4:
		// Change activation of one neuron
		bp.ChangeActivationFunction()
	}
	// Additional random clamp
	autoClampIfInf(bp, minClamp, maxClamp)
}

// pruneLowWeightNeurons – example function removing neurons whose total connection magnitude is too small
func pruneLowWeightNeurons(bp *phase.Phase) {
	const threshold = 0.001 // if sum of absolute weights < threshold, remove
	var toDelete []int
	for id, neuron := range bp.Neurons {
		if neuron.Type == "input" || contains(bp.OutputNodes, id) {
			continue
		}
		sumW := 0.0
		for _, conn := range neuron.Connections {
			sumW += math.Abs(conn[1])
		}
		if math.Abs(neuron.Bias) > 0 {
			sumW += math.Abs(neuron.Bias)
		}
		if sumW < threshold {
			toDelete = append(toDelete, id)
		}
	}
	if len(toDelete) > 0 {
		for _, id := range toDelete {
			delete(bp.Neurons, id)
		}
		fmt.Printf("   => Pruned %d low-weight neurons.\n", len(toDelete))
	}
}

// crossoverPhases merges two parent Phases
func crossoverPhases(pA, pB *phase.Phase) *phase.Phase {
	return phaseCrossoverSimple(pA, pB)
}

// phaseCrossoverSimple – a simple approach that merges weights from each parent
func phaseCrossoverSimple(a, b *phase.Phase) *phase.Phase {
	offspring := a.Copy()
	for id, neuronB := range b.Neurons {
		if neuronA, ok := offspring.Neurons[id]; ok {
			// Merge weights – random choice or average
			for i := range neuronA.Connections {
				if i < len(neuronB.Connections) {
					if rand.Float64() < 0.5 {
						neuronA.Connections[i][1] = neuronB.Connections[i][1]
					} else {
						// average
						neuronA.Connections[i][1] = 0.5 * (neuronA.Connections[i][1] + neuronB.Connections[i][1])
					}
				}
			}
			// random pick or average bias
			if rand.Float64() < 0.5 {
				neuronA.Bias = neuronB.Bias
			} else {
				neuronA.Bias = 0.5 * (neuronA.Bias + neuronB.Bias)
			}
		} else {
			// Possibly clone neuron from B
			if rand.Float64() < 0.3 {
				offspring.Neurons[id] = deepCopyNeuron(neuronB)
			}
		}
	}
	autoClampIfInf(offspring, minClamp, maxClamp)
	return offspring
}

// deepCopyNeuron used in crossover
func deepCopyNeuron(n *phase.Neuron) *phase.Neuron {
	if n == nil {
		return nil
	}
	n2 := &phase.Neuron{
		ID:               n.ID,
		Type:             n.Type,
		Value:            n.Value,
		Bias:             n.Bias,
		Activation:       n.Activation,
		WindowSize:       n.WindowSize,
		DropoutRate:      n.DropoutRate,
		BatchNorm:        n.BatchNorm,
		Attention:        n.Attention,
		CellState:        n.CellState,
		BatchNormParams:  n.BatchNormParams,
		GateWeights:      nil,
		Connections:      nil,
		AttentionWeights: nil,
		Kernels:          nil,
		LoopCount:        n.LoopCount,
		NCAState:         nil,
		NeighborhoodIDs:  nil,
		UpdateRules:      n.UpdateRules,
	}
	// Copy connections
	n2.Connections = make([][]float64, len(n.Connections))
	for i, c := range n.Connections {
		n2.Connections[i] = []float64{c[0], c[1]}
	}
	// Copy gate weights
	if n.GateWeights != nil {
		tmp := make(map[string][]float64)
		for k, arr := range n.GateWeights {
			cp := make([]float64, len(arr))
			copy(cp, arr)
			tmp[k] = cp
		}
		n2.GateWeights = tmp
	}
	// Copy attention weights
	if n.AttentionWeights != nil {
		cp2 := make([]float64, len(n.AttentionWeights))
		copy(cp2, n.AttentionWeights)
		n2.AttentionWeights = cp2
	}
	// Copy kernels
	if len(n.Kernels) > 0 {
		n2.Kernels = make([][]float64, len(n.Kernels))
		for i := range n.Kernels {
			cp3 := make([]float64, len(n.Kernels[i]))
			copy(cp3, n.Kernels[i])
			n2.Kernels[i] = cp3
		}
	}
	// Copy NCAState
	if len(n.NCAState) > 0 {
		tmp4 := make([]float64, len(n.NCAState))
		copy(tmp4, n.NCAState)
		n2.NCAState = tmp4
	}
	// Copy neighbor IDs
	if len(n.NeighborhoodIDs) > 0 {
		tmp5 := make([]int, len(n.NeighborhoodIDs))
		copy(tmp5, n.NeighborhoodIDs)
		n2.NeighborhoodIDs = tmp5
	}
	return n2
}

// ============================================================================
// Step 4: Evaluate & Accuracy
// ============================================================================

func evaluateAccuracy(bp *phase.Phase, X, Y *mat.Dense) (exactAcc, closeAcc, proximityScore float64) {
	nSamples, _ := X.Dims()
	if nSamples == 0 {
		return 0, 0, 0
	}
	correctExact := 0
	totalClose := 0.0
	totalProx := 0.0

	autoClampIfInf(bp, minClamp, maxClamp)
	sampleContribution := 100.0 / float64(nSamples)

	for i := 0; i < nSamples; i++ {
		inputs := make(map[int]float64)
		for px := 0; px < 784; px++ {
			inputs[bp.InputNodes[px]] = X.At(i, px)
		}
		bp.RunNetwork(inputs, 1)

		vals := make([]float64, 10)
		for j := 0; j < 10; j++ {
			val := bp.Neurons[bp.OutputNodes[j]].Value
			if math.IsNaN(val) || math.IsInf(val, 0) {
				val = 0
			}
			vals[j] = val
		}

		maxVal := max(vals)
		if math.IsNaN(maxVal) || maxVal <= 0 {
			maxVal = 1.0
		}
		actual := int(Y.At(i, 0))
		pred := argmax(vals)
		correctVal := vals[actual]

		if pred == actual {
			correctExact++
			totalProx += sampleContribution
			totalClose += 1.0
		} else {
			proximityRatio := 0.0
			closeRatio := 0.0
			if correctVal >= 0 && maxVal > 0 {
				if correctVal <= maxVal {
					proximityRatio = correctVal / maxVal
				} else if correctVal <= 2.0*maxVal {
					pr := 1.0 - (correctVal-maxVal)/maxVal
					if pr < 0 {
						pr = 0.0
					}
					proximityRatio = pr
				}
				lower := 0.9 * maxVal
				upper := 1.1 * maxVal
				if correctVal >= lower && correctVal <= upper {
					if correctVal <= maxVal {
						closeRatio = (correctVal - lower) / (maxVal - lower)
					} else {
						closeRatio = (upper - correctVal) / (upper - maxVal)
					}
					if closeRatio < 0 {
						closeRatio = 0.0
					}
				}
			}
			totalProx += (proximityRatio * sampleContribution)
			totalClose += closeRatio
		}
	}
	exactAcc = float64(correctExact) / float64(nSamples)
	closeAcc = totalClose / float64(nSamples)
	proximityScore = totalProx
	if math.IsNaN(proximityScore) || math.IsInf(proximityScore, 0) {
		proximityScore = 0
	}
	if math.IsNaN(closeAcc) || math.IsInf(closeAcc, 0) {
		closeAcc = 0
	}
	return exactAcc, closeAcc, proximityScore
}

// ============================================================================
// Step 5: Fine-Tuning Tools – Adjustments, etc.
// ============================================================================

func generateAdjustments(min, max, step float64) []float64 {
	var out []float64
	for val := min; val <= max; val += step {
		out = append(out, val)
	}
	return out
}

func findBestAdjustment(model *phase.Phase, adjustments []float64, testX, testY *mat.Dense, isRushMode bool) (*phase.Phase, float64, float64) {
	origExact, origClose, origProx := evaluateAccuracy(model, testX, testY)
	bestFit := 0.5*origExact + 0.3*origClose + 0.2*(origProx/100)
	bestProx := origProx
	bestModel := model.Copy()

	var tasks []AdjustmentTask
	for _, a := range adjustments {
		tasks = append(tasks,
			AdjustmentTask{true, false, a},
			AdjustmentTask{false, true, a},
			AdjustmentTask{true, true, a},
		)
	}

	// concurrency
	numWorkers := int(math.Floor(float64(runtime.NumCPU()) * 0.8))
	if numWorkers < 1 {
		numWorkers = 1
	}
	semaphore := make(chan struct{}, numWorkers)
	var wg sync.WaitGroup
	results := make(chan AdjustmentResult, len(tasks))

	for _, t := range tasks {
		wg.Add(1)
		semaphore <- struct{}{}
		go func(task AdjustmentTask) {
			defer wg.Done()
			defer func() { <-semaphore }()

			cand := model.Copy()
			if task.adjustWeights {
				cand.AdjustAllWeights(task.adjValue)
			}
			if task.adjustBiases {
				cand.AdjustAllBiases(task.adjValue)
			}
			ex, cl, prox := evaluateAccuracy(cand, testX, testY)
			f := 0.5*ex + 0.3*cl + 0.2*(prox/100)
			results <- AdjustmentResult{model: cand, fitness: f, proximityScore: prox}
		}(t)
	}
	wg.Wait()
	close(results)

	for r := range results {
		if isRushMode {
			if r.proximityScore > bestProx {
				bestProx = r.proximityScore
				bestFit = r.fitness
				bestModel = r.model
			}
		} else {
			if r.fitness > bestFit {
				bestFit = r.fitness
				bestProx = r.proximityScore
				bestModel = r.model
			}
		}
	}
	return bestModel, bestFit, bestProx
}

// ============================================================================
// Step 6: Additional
// ============================================================================

func updateKnownSamples(bp *phase.Phase, testX, testY *mat.Dense, known map[int]bool) {
	r, _ := testX.Dims()
	for i := 0; i < r; i++ {
		inputs := make(map[int]float64)
		for px := 0; px < 784; px++ {
			inputs[bp.InputNodes[px]] = testX.At(i, px)
		}
		bp.RunNetwork(inputs, 1)
		vals := make([]float64, 10)
		for j := 0; j < 10; j++ {
			vals[j] = bp.Neurons[bp.OutputNodes[j]].Value
		}
		if argmax(vals) == int(testY.At(i, 0)) {
			known[i] = true
		} else {
			known[i] = false
		}
	}
}

func finishAndExit(pop []*phase.Phase, gen int, bestFitness float64, compLevel int, stag int, bestModel *phase.Phase,
	trainX, trainY, testX, testY *mat.Dense, knownSamples map[int]bool) {
	st := EvolutionaryState{
		Population:        pop,
		Generation:        gen,
		BestFitness:       bestFitness,
		ComplexityLevel:   compLevel,
		StagnationCounter: stag,
		KnownSamples:      knownSamples,
		BestModel:         bestModel,
	}
	if err := saveState(st, stateFile); err != nil {
		log.Printf("Failed saving state: %v", err)
	}

	trEx, trCl, trPx := evaluateAccuracy(bestModel, trainX, trainY)
	tsEx, tsCl, tsPx := evaluateAccuracy(bestModel, testX, testY)
	res := Results{
		TrainExactAcc:       trEx,
		TrainCloseAcc:       trCl,
		TrainProximityScore: trPx,
		TestExactAcc:        tsEx,
		TestCloseAcc:        tsCl,
		TestProximityScore:  tsPx,
	}
	if err := saveResults(res, resultsFile); err != nil {
		log.Printf("Failed to save results: %v", err)
	}
	visualizeResults(bestModel, trainX, trainY, testX, testY)
	os.Exit(0)
}

// ============================================================================
// Utilities: Argmax, MaxFitness, etc.
// ============================================================================

func max(vals []float64) float64 {
	m := vals[0]
	for _, v := range vals {
		if !math.IsNaN(v) && !math.IsInf(v, 0) && v > m {
			m = v
		}
	}
	return m
}

func argmax(vals []float64) int {
	idx := 0
	maxV := vals[0]
	for i := 1; i < len(vals); i++ {
		if !math.IsNaN(vals[i]) && !math.IsInf(vals[i], 0) && vals[i] > maxV {
			maxV = vals[i]
			idx = i
		}
	}
	return idx
}

func maxFitness(fit []float64) (float64, int) {
	best := fit[0]
	bestI := 0
	for i, f := range fit {
		val := f
		if math.IsNaN(val) || math.IsInf(val, 0) {
			val = 0
		}
		if val > best {
			best = val
			bestI = i
		}
	}
	return best, bestI
}

func averageMinFitness(fit []float64) (float64, float64) {
	if len(fit) == 0 {
		return 0, 0
	}
	sum := 0.0
	minVal := fit[0]
	for _, f := range fit {
		val := f
		if math.IsNaN(val) || math.IsInf(val, 0) {
			val = 0
		}
		sum += val
		if val < minVal {
			minVal = val
		}
	}
	return sum / float64(len(fit)), minVal
}

// selectTopPerformers sorts by exact -> close -> prox
func selectTopPerformers(pop []*phase.Phase, exacts, closes, prox []float64, fraction float64) []*phase.Phase {
	n := len(pop)
	num := int(float64(n) * fraction)
	if num < 1 {
		num = 1
	}
	type scored struct {
		idx  int
		eAcc float64
		cAcc float64
		pAcc float64
	}
	arr := make([]scored, n)
	for i := 0; i < n; i++ {
		arr[i] = scored{i, exacts[i], closes[i], prox[i]}
	}
	sort.Slice(arr, func(a, b int) bool {
		aa, bb := arr[a], arr[b]
		if aa.eAcc != bb.eAcc {
			return aa.eAcc > bb.eAcc
		}
		if aa.cAcc != bb.cAcc {
			return aa.cAcc > bb.cAcc
		}
		return aa.pAcc > bb.pAcc
	})
	selected := make([]*phase.Phase, num)
	for i := 0; i < num; i++ {
		selected[i] = pop[arr[i].idx].Copy()
	}
	return selected
}

// ============================================================================
// Save/Load, Visualization
// ============================================================================

func loadState(fname string) (EvolutionaryState, error) {
	var st EvolutionaryState
	data, err := os.ReadFile(filepath.Join(saveFolder, fname))
	if err != nil {
		return st, err
	}
	err = json.Unmarshal(data, &st)
	return st, err
}

func saveState(st EvolutionaryState, fname string) error {
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(saveFolder, fname), data, 0644)
}

func saveModel(bp *phase.Phase, generation int, fitness float64, improved bool) error {
	data, err := json.MarshalIndent(bp, "", "  ")
	if err != nil {
		return err
	}
	tag := "model"
	if improved {
		tag = "improved_model"
	}
	filename := fmt.Sprintf("%s_gen_%d_fitness_%.4f.json", tag, generation, fitness)
	return os.WriteFile(filepath.Join(saveFolder, filename), data, 0644)
}

func saveResults(r Results, fname string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(saveFolder, fname), data, 0644)
}

// autoClampIfInf forcibly clamps extreme or NaN values
func autoClampIfInf(bp *phase.Phase, minVal, maxVal float64) {
	needsClamp := false
	for _, neuron := range bp.Neurons {
		if math.IsNaN(neuron.Value) || math.IsInf(neuron.Value, 0) {
			needsClamp = true
			break
		}
	}
	if !needsClamp {
		return
	}
	fmt.Println("Auto clamp triggered. Clamping to range [", minVal, ",", maxVal, "]")

	numWorkers := runtime.NumCPU()
	ids := make([]int, 0, len(bp.Neurons))
	for id := range bp.Neurons {
		ids = append(ids, id)
	}
	chunkSize := (len(ids) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(ids) {
			end = len(ids)
		}
		if start >= len(ids) {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				id := ids[i]
				n := bp.Neurons[id]
				// Value
				if math.IsNaN(n.Value) || math.IsInf(n.Value, 0) {
					n.Value = 0
				}
				if n.Value > maxVal {
					n.Value = maxVal
				} else if n.Value < minVal {
					n.Value = minVal
				}
				// Cell
				if math.IsNaN(n.CellState) || math.IsInf(n.CellState, 0) {
					n.CellState = 0
				}
				if n.CellState > maxVal {
					n.CellState = maxVal
				} else if n.CellState < minVal {
					n.CellState = minVal
				}
				// Bias
				if math.IsNaN(n.Bias) || math.IsInf(n.Bias, 0) {
					n.Bias = 0
				}
				if n.Bias > maxVal {
					n.Bias = maxVal
				} else if n.Bias < minVal {
					n.Bias = minVal
				}
				// Connections
				for j := range n.Connections {
					wVal := n.Connections[j][1]
					if math.IsNaN(wVal) || math.IsInf(wVal, 0) {
						wVal = 0
					}
					if wVal > maxVal {
						wVal = maxVal
					} else if wVal < minVal {
						wVal = minVal
					}
					n.Connections[j][1] = wVal
				}
			}
		}(start, end)
	}
	wg.Wait()
}

// Visualization
func visualizeResults(bp *phase.Phase, trainX, trainY, testX, testY *mat.Dense) {
	fmt.Println("\n--- Quick Visualization ---")
	trainRows, _ := trainX.Dims()
	fmt.Println("Train samples (first 5):")
	for i := 0; i < 5 && i < trainRows; i++ {
		inputs := make(map[int]float64)
		for px := 0; px < 784; px++ {
			inputs[bp.InputNodes[px]] = trainX.At(i, px)
		}
		bp.RunNetwork(inputs, 1)
		vals := make([]float64, 10)
		for j := 0; j < 10; j++ {
			vals[j] = bp.Neurons[bp.OutputNodes[j]].Value
		}
		pred := argmax(vals)
		actual := int(trainY.At(i, 0))
		fmt.Printf("  Train Sample %d => pred=%d, actual=%d\n", i, pred, actual)
	}

	testRows, _ := testX.Dims()
	fmt.Println("Test samples (first 5):")
	for i := 0; i < 5 && i < testRows; i++ {
		inputs := make(map[int]float64)
		for px := 0; px < 784; px++ {
			inputs[bp.InputNodes[px]] = testX.At(i, px)
		}
		bp.RunNetwork(inputs, 1)
		vals := make([]float64, 10)
		for j := 0; j < 10; j++ {
			vals[j] = bp.Neurons[bp.OutputNodes[j]].Value
		}
		pred := argmax(vals)
		actual := int(testY.At(i, 0))
		fmt.Printf("  Test Sample %d => pred=%d, actual=%d\n", i, pred, actual)
	}

	trainExact, trainClose, trainProx := evaluateAccuracy(bp, trainX, trainY)
	testExact, testClose, testProx := evaluateAccuracy(bp, testX, testY)

	fmt.Printf("\nFinal Train Exact=%.2f%%, Close=%.2f%%, Prox=%.2f%%\n",
		trainExact*100, trainClose*100, trainProx)
	fmt.Printf("Final Test  Exact=%.2f%%, Close=%.2f%%, Prox=%.2f%%\n\n",
		testExact*100, testClose*100, testProx)
}

// contains is a helper
func contains(arr []int, v int) bool {
	for _, x := range arr {
		if x == v {
			return true
		}
	}
	return false
}

func getNeuronRangeByComplexity(complexity int) (int, int) {
	switch complexity {
	case 1:
		return Complexity1MinNeurons, Complexity1MaxNeurons
	case 2:
		return Complexity2MinNeurons, Complexity2MaxNeurons
	case 3:
		return Complexity3MinNeurons, Complexity3MaxNeurons
	default:
		minN := 10 * complexity
		maxN := 50 * complexity
		return minN, maxN
	}
}
