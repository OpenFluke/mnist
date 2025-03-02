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

	"phase" // Adjust this import path to your local "phase" package if needed.
)

// ============================================================================
// Constants & Types
// ============================================================================

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"

	// Evolution hyperparameters
	populationSize      = 50
	numGenerations      = 5000
	selectionPercentage = 0.3
	mutationRate        = 0.3

	// Training hyperparameters
	learningRate = 0.01
	trainEpochs  = 1

	// For saving state
	saveFolder  = "test"
	stateFile   = "nas_population.json"
	resultsFile = "results.json"

	stagnationCounterMax  = 5
	Complexity1MinNeurons = 5
	Complexity1MaxNeurons = 30
	Complexity2MinNeurons = 50
	Complexity2MaxNeurons = 80
	Complexity3MinNeurons = 100
	Complexity3MaxNeurons = 500

	maxClamp = 1000
	minClamp = -1000

	closeThreshold = 0.9

	// Default split: 80% training, 20% testing
	trainPercentage = 0.8

	// Connection settings for new neurons
	minConnections = 10  // Minimum incoming connections per new neuron
	maxConnections = 150 // Maximum incoming connections per new neuron

	rushModeEnabled = true
	rushThreshold   = 0.6
)

// EvolutionaryState holds the overall evolutionary process state
type EvolutionaryState struct {
	Population        []*phase.Phase
	Generation        int
	BestFitness       float64
	ComplexityLevel   int
	StagnationCounter int
	KnownSamples      map[int]bool // Tracks indices of correctly classified test samples
	BestModel         *phase.Phase // Save the best model
}

// Results holds final stats for all metrics
type Results struct {
	TrainExactAcc       float64
	TrainCloseAcc       float64
	TrainProximityScore float64
	TestExactAcc        float64
	TestCloseAcc        float64
	TestProximityScore  float64
}

// BestModelTracker tracks the best model with synchronization
type BestModelTracker struct {
	mu                 sync.Mutex
	overallBestModel   *phase.Phase
	bestExactAcc       float64
	bestCloseAcc       float64
	bestProximityScore float64
	updated            bool
}

// ============================================================================
// Main
// ============================================================================

func main() {
	// Maximize CPU usage and print the change
	prev := runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Printf("Previous GOMAXPROCS: %d, now set to %d\n", prev, runtime.NumCPU())

	rand.Seed(time.Now().UnixNano())

	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Load full training dataset (60,000 images)
	trainX, trainY, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading full training MNIST: %v", err)
	}
	// Load full testing dataset (10,000 images)
	testX, testY, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading full testing MNIST: %v", err)
	}

	// Combine and shuffle dataset
	allX := mat.NewDense(70000, 784, nil)
	allX.Stack(trainX, testX)
	allY := mat.NewDense(70000, 1, nil)
	allY.Stack(trainY, testY)

	perm := rand.Perm(70000)
	shuffledX := mat.NewDense(70000, 784, nil)
	shuffledY := mat.NewDense(70000, 1, nil)
	for i, p := range perm {
		shuffledX.SetRow(i, allX.RawRowView(p))
		shuffledY.Set(i, 0, allY.At(p, 0))
	}

	// Split into training and testing sets
	trainSamples := int(float64(70000) * trainPercentage)
	testSamples := 70000 - trainSamples

	newTrainX := mat.NewDense(trainSamples, 784, nil)
	newTrainY := mat.NewDense(trainSamples, 1, nil)
	newTestX := mat.NewDense(testSamples, 784, nil)
	newTestY := mat.NewDense(testSamples, 1, nil)

	newTrainX.Copy(shuffledX.Slice(0, trainSamples, 0, 784))
	newTrainY.Copy(shuffledY.Slice(0, trainSamples, 0, 1))
	newTestX.Copy(shuffledX.Slice(trainSamples, 70000, 0, 784))
	newTestY.Copy(shuffledY.Slice(trainSamples, 70000, 0, 1))

	// Train and test
	trainTesting(newTrainX, newTrainY, newTestX, newTestY)
	fmt.Println("Done.")
}

// ============================================================================
// Helper Functions
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
// Evolutionary Training
// ============================================================================

func trainTesting(trainX, trainY, testX, testY *mat.Dense) {
	fmt.Println(time.Now())

	if err := os.MkdirAll(saveFolder, 0755); err != nil {
		log.Fatalf("Failed to create folder %s: %v", saveFolder, err)
	}

	// Calculate 80% of CPU cores for parallel processing
	cpuCount := runtime.NumCPU()
	targetCores := int(math.Floor(float64(cpuCount) * 0.8))
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

	// Load saved state if available
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
		if bestModel == nil {
			bestModel = population[0].Copy() // Fallback to first model if bestModel is nil
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
		fmt.Printf("\n=== Generation %d (Complexity Level: %d) ===\n", gen, complexityLevel)
		fmt.Println(time.Now())

		fitness := make([]float64, populationSize)
		exactAccs := make([]float64, populationSize)
		closeAccs := make([]float64, populationSize)
		proximityScores := make([]float64, populationSize)

		bestModelTracker.mu.Lock()
		bestModelTracker.updated = false
		bestModelTracker.mu.Unlock()

		// Evaluate population using 80% of CPU cores
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, targetCores)
		for i := 0; i < populationSize; i++ {
			wg.Add(1)
			semaphore <- struct{}{} // Acquire semaphore
			go func(idx int) {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore
				bp := population[idx]
				exactAcc, closeAcc, proximityScore := evaluateAccuracy(bp, testX, testY)
				fitness[idx] = (0.5 * exactAcc) + (0.3 * closeAcc) + (0.2 * proximityScore / 100)
				exactAccs[idx] = exactAcc
				closeAccs[idx] = closeAcc
				proximityScores[idx] = proximityScore

				fmt.Printf("     Network %d: exact=%.2f%%, close=%.2f%%, proximity=%.2f%%\n",
					idx, exactAcc*100, closeAcc*100, proximityScore)

				bestModelTracker.mu.Lock()
				updateReason := ""
				if rushModeEnabled && bestModelTracker.bestExactAcc < rushThreshold {
					if proximityScore > bestModelTracker.bestProximityScore {
						updateReason = "proximity score improved (rush mode)"
					}
				} else {
					if exactAcc > bestModelTracker.bestExactAcc {
						updateReason = "exact accuracy improved"
					} else if exactAcc == bestModelTracker.bestExactAcc && closeAcc > bestModelTracker.bestCloseAcc {
						updateReason = "close accuracy improved"
					} else if exactAcc == bestModelTracker.bestExactAcc && closeAcc == bestModelTracker.bestCloseAcc && proximityScore > bestModelTracker.bestProximityScore {
						updateReason = "proximity score improved"
					}
				}
				if updateReason != "" {
					bestModelTracker.overallBestModel = bp.Copy()
					bestModelTracker.bestExactAcc = exactAcc
					bestModelTracker.bestCloseAcc = closeAcc
					bestModelTracker.bestProximityScore = proximityScore
					bestModelTracker.updated = true
					fmt.Printf("     => New best model selected (%s): exact=%.2f%%, close=%.2f%%, proximity=%.2f%%\n",
						updateReason, exactAcc*100, closeAcc*100, proximityScore)
				}
				bestModelTracker.mu.Unlock()
			}(i)
		}
		wg.Wait()

		currentBest, bestIndex := maxFitness(fitness)
		fmt.Printf("Best fitness this generation: %.4f (exact=%.2f%%, close=%.2f%%, proximity=%.2f%%)\n",
			currentBest, exactAccs[bestIndex]*100, closeAccs[bestIndex]*100, proximityScores[bestIndex])

		if err := saveModel(population[bestIndex], gen, currentBest, currentBest > bestFitness); err != nil {
			log.Printf("Failed to save model (gen %d): %v", gen, err)
		}

		if currentBest >= 0.9999 {
			bestFitness = currentBest
			if bestModelTracker.overallBestModel == nil {
				bestModelTracker.overallBestModel = population[bestIndex].Copy()
			}
			fmt.Printf("Reached near-perfect fitness; stopping at generation %d.\n", gen)
			updateKnownSamples(bestModelTracker.overallBestModel, testX, testY, knownSamples)
			finishAndExit(population, gen+1, bestFitness, complexityLevel, stagnationCounter,
				bestModelTracker.overallBestModel, trainX, trainY, testX, testY, knownSamples)
			return
		}

		updateKnownSamples(bestModelTracker.overallBestModel, testX, testY, knownSamples)

		// Create next generation
		selected := selectTopPerformers(population, exactAccs, closeAccs, proximityScores, selectionPercentage)
		population = createNextGeneration(selected, populationSize, mutationRate, complexityLevel,
			bestModelTracker.overallBestModel, knownSamples, testX, testY)

		bestModelTracker.mu.Lock()
		if bestModelTracker.updated {
			bestFitness = currentBest
			stagnationCounter = 0
			fmt.Printf("  => Improvement! New best model with exact=%.2f%%, close=%.2f%%, proximity=%.2f%%\n",
				bestModelTracker.bestExactAcc*100, bestModelTracker.bestCloseAcc*100, bestModelTracker.bestProximityScore)
		} else {
			stagnationCounter++
			fmt.Printf("  => No improvement. Stagnation %d/%d\n", stagnationCounter, stagnationCounterMax)
			if stagnationCounter >= stagnationCounterMax {
				complexityLevel++
				stagnationCounter = 0
				fmt.Printf("  => Complexity raised to %d\n", complexityLevel)
			}
		}
		bestModelTracker.mu.Unlock()

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

		gen++
	}

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
	fmt.Println(time.Now())
}

func finishAndExit(pop []*phase.Phase, g int, bestFitness float64, comp int, stag int, bestModel *phase.Phase,
	trainX, trainY, testX, testY *mat.Dense, knownSamples map[int]bool) {
	state := EvolutionaryState{
		Population:        pop,
		Generation:        g,
		BestFitness:       bestFitness,
		ComplexityLevel:   comp,
		StagnationCounter: stag,
		KnownSamples:      knownSamples,
		BestModel:         bestModel,
	}
	if err := saveState(state, stateFile); err != nil {
		log.Printf("Failed saving state: %v", err)
	}
	trainExact, trainClose, trainProx := evaluateAccuracy(bestModel, trainX, trainY)
	testExact, testClose, testProx := evaluateAccuracy(bestModel, testX, testY)
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
	visualizeResults(bestModel, trainX, trainY, testX, testY)
}

// ============================================================================
// Evolution Utilities
// ============================================================================

func createInitialPopulation(popSize int) []*phase.Phase {
	pop := make([]*phase.Phase, popSize)
	for i := 0; i < popSize; i++ {
		bp := phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
		pop[i] = bp
	}
	return pop
}

func trainNetwork(bp *phase.Phase, X, Y *mat.Dense, epochs int) {
	nSamples, _ := X.Dims()
	for e := 0; e < epochs; e++ {
		perm := rand.Perm(nSamples)
		for _, i := range perm {
			inputs := make(map[int]float64)
			for px := 0; px < 784; px++ {
				inputs[bp.InputNodes[px]] = X.At(i, px)
			}
			label := int(Y.At(i, 0))
			expected := make(map[int]float64)
			for out := 0; out < 10; out++ {
				expected[bp.OutputNodes[out]] = 0.0
			}
			expected[bp.OutputNodes[label]] = 1.0

			bp.Forward(inputs, 1)
			// Uncomment and adjust training method as needed
			// bp.TrainNetwork(inputs, expected, learningRate, minClamp, maxClamp)
		}
		autoClampIfInf(bp, float64(minClamp), float64(maxClamp))
	}
}

func evaluateAccuracy(bp *phase.Phase, X, Y *mat.Dense) (exactAcc, closeAcc, proximityScore float64) {
	nSamples, _ := X.Dims()
	correctExact := 0
	totalClose := 0.0
	totalProximity := 0.0

	autoClampIfInf(bp, float64(minClamp), float64(maxClamp))

	sampleContribution := 100.0 / float64(nSamples)
	for i := 0; i < nSamples; i++ {
		inputs := make(map[int]float64)
		for px := 0; px < 784; px++ {
			inputs[bp.InputNodes[px]] = X.At(i, px)
		}
		bp.RunNetwork(inputs, 1)

		vals := make([]float64, 10)
		for j := 0; j < 10; j++ {
			vals[j] = bp.Neurons[bp.OutputNodes[j]].Value
			if math.IsNaN(vals[j]) || math.IsInf(vals[j], 0) {
				vals[j] = 0.0
			}
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
			totalProximity += sampleContribution
			totalClose += 1.0
		} else {
			proximityRatio := 0.0
			closeRatio := 0.0
			if !math.IsNaN(correctVal) && correctVal >= 0 && maxVal > 0 {
				if correctVal <= maxVal {
					proximityRatio = correctVal / maxVal
				} else if correctVal <= 2.0*maxVal {
					proximityRatio = 1.0 - (correctVal-maxVal)/maxVal
					if proximityRatio < 0 {
						proximityRatio = 0.0
					}
				}

				lowerBound := 0.9 * maxVal
				upperBound := 1.1 * maxVal
				if correctVal >= lowerBound && correctVal <= upperBound {
					if correctVal <= maxVal {
						closeRatio = (correctVal - lowerBound) / (maxVal - lowerBound)
					} else {
						closeRatio = (upperBound - correctVal) / (upperBound - maxVal)
					}
					if closeRatio < 0 {
						closeRatio = 0.0
					}
				}
			}
			totalProximity += proximityRatio * sampleContribution
			totalClose += closeRatio
		}
	}

	exactAcc = float64(correctExact) / float64(nSamples)
	closeAcc = totalClose / float64(nSamples)
	proximityScore = totalProximity
	if math.IsNaN(proximityScore) || math.IsInf(proximityScore, 0) {
		proximityScore = 0.0
	}
	if math.IsNaN(closeAcc) || math.IsInf(closeAcc, 0) {
		closeAcc = 0.0
	}
	return exactAcc, closeAcc, proximityScore
}

func max(vals []float64) float64 {
	maxVal := vals[0]
	for _, v := range vals {
		if v > maxVal && !math.IsNaN(v) && !math.IsInf(v, 0) {
			maxVal = v
		}
	}
	return maxVal
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

func mutate(bp *phase.Phase, mRate float64, complexity int) {
	beforeNeurons := make(map[int]struct{}, len(bp.Neurons))
	for id := range bp.Neurons {
		beforeNeurons[id] = struct{}{}
	}

	switch rand.Intn(12) {
	case 0:
		nMin, nMax := getNeuronRangeByComplexity(complexity)
		if nMax < nMin {
			nMax = nMin
		}
		numNewNeurons := rand.Intn(nMax-nMin+1) + nMin
		fmt.Printf("  => Complexity %d, adding %d new neurons\n", complexity, numNewNeurons)
		for i := 0; i < numNewNeurons; i++ {
			bp.AddRandomNeuron("", "", complexity+minConnections, maxConnections+complexity+2)
			bp.RewireOutputsThroughNewNeuron(bp.GetNextNeuronID() - 1)
		}
	case 1:
		bp.AddConnection()
	case 2:
		bp.RemoveConnection()
	case 3:
		bp.AdjustWeights()
	case 4:
		bp.AdjustBiases()
	case 5:
		bp.ChangeActivationFunction()
	case 6:
		bp.AdjustAllWeights(rand.NormFloat64() * 0.01)
	case 7:
		bp.AdjustAllBiases(rand.NormFloat64() * 0.01)
	case 8:
		bp.ChangeSingleNeuronType()
	case 9:
		bp.ChangePercentageOfNeuronsTypes(10.0)
	case 10:
		bp.RandomizeAllNeuronsTypes()
	case 11:
		bp.SetAllNeuronsToSameRandomType()
	}

	afterNeurons := make([]int, 0, len(bp.Neurons))
	for id := range bp.Neurons {
		afterNeurons = append(afterNeurons, id)
	}

	newNeurons := []int{}
	for _, id := range afterNeurons {
		if _, existed := beforeNeurons[id]; !existed {
			newNeurons = append(newNeurons, id)
		}
	}
	bp.TrainableNeurons = newNeurons
}

func selectTopPerformers(pop []*phase.Phase, exactAccs, closeAccs, proximityScores []float64, frac float64) []*phase.Phase {
	numSelect := int(float64(len(pop)) * frac)
	type modelScore struct {
		idx            int
		exactAcc       float64
		closeAcc       float64
		proximityScore float64
	}
	scores := make([]modelScore, len(pop))
	for i := 0; i < len(pop); i++ {
		scores[i] = modelScore{i, exactAccs[i], closeAccs[i], proximityScores[i]}
	}
	sort.Slice(scores, func(a, b int) bool {
		if scores[a].exactAcc != scores[b].exactAcc {
			return scores[a].exactAcc > scores[b].exactAcc
		}
		if scores[a].closeAcc != scores[b].closeAcc {
			return scores[a].closeAcc > scores[b].closeAcc
		}
		return scores[a].proximityScore > scores[b].proximityScore
	})
	selected := make([]*phase.Phase, numSelect)
	for i := 0; i < numSelect; i++ {
		selected[i] = pop[scores[i].idx].Copy()
	}
	return selected
}

func createNextGeneration(parents []*phase.Phase, popSize int, mRate float64, cLevel int, overallBestModel *phase.Phase, knownSamples map[int]bool, testX, testY *mat.Dense) []*phase.Phase {
	out := make([]*phase.Phase, popSize)
	out[0] = overallBestModel.Copy()

	targetedFraction := 0.1
	numTargeted := int(math.Round(float64(popSize-1) * targetedFraction))
	targetedStart := popSize - numTargeted

	for i := 1; i < targetedStart; i++ {
		p := parents[rand.Intn(len(parents))]
		child := p.Copy()
		mutate(child, mRate, cLevel)
		out[i] = child
	}

	nSamples, _ := testX.Dims()
	unknownIndices := make([]int, 0, nSamples)
	for i := 0; i < nSamples; i++ {
		if !knownSamples[i] {
			unknownIndices = append(unknownIndices, i)
		}
	}

	for i := targetedStart; i < popSize; i++ {
		if len(unknownIndices) == 0 {
			p := parents[rand.Intn(len(parents))]
			child := p.Copy()
			mutate(child, mRate, cLevel)
			out[i] = child
			continue
		}

		idx := unknownIndices[rand.Intn(len(unknownIndices))]
		bestChild := performTargetedMutation(overallBestModel, idx, testX, testY, mRate, cLevel)
		originalExact, _, _ := evaluateAccuracy(overallBestModel, testX, testY)
		newExact, _, _ := evaluateAccuracy(bestChild, testX, testY)
		if newExact >= originalExact {
			out[i] = bestChild
		} else {
			out[i] = overallBestModel.Copy()
		}
	}

	return out
}

func performTargetedMutation(bp *phase.Phase, sampleIdx int, testX, testY *mat.Dense, mRate float64, cLevel int) *phase.Phase {
	inputs := make(map[int]float64)
	for px := 0; px < 784; px++ {
		inputs[bp.InputNodes[px]] = testX.At(sampleIdx, px)
	}
	actual := int(testY.At(sampleIdx, 0))

	type MutationResult struct {
		model *phase.Phase
		score float64
	}

	// Use 80% of CPU cores
	cpuCount := runtime.NumCPU()
	numGoroutines := int(math.Floor(float64(cpuCount) * 0.8))
	if numGoroutines < 1 {
		numGoroutines = 1
	}

	results := make(chan MutationResult, numGoroutines)
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			candidate := bp.Copy()
			mutate(candidate, mRate, cLevel)
			candidate.RunNetwork(inputs, 1)
			newVals := make([]float64, 10)
			for k := 0; k < 10; k++ {
				newVals[k] = candidate.Neurons[candidate.OutputNodes[k]].Value
			}
			newScore := newVals[actual]
			results <- MutationResult{candidate, newScore}
		}()
	}

	wg.Wait()
	close(results)

	bestScore := math.Inf(-1)
	var bestModel *phase.Phase
	for res := range results {
		if res.score > bestScore {
			bestScore = res.score
			bestModel = res.model
		}
	}
	if bestModel == nil {
		return bp.Copy()
	}
	return bestModel
}

func updateKnownSamples(bp *phase.Phase, testX, testY *mat.Dense, knownSamples map[int]bool) {
	nSamples, _ := testX.Dims()
	for i := 0; i < nSamples; i++ {
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
		knownSamples[i] = pred == actual
	}
}

func maxFitness(fit []float64) (float64, int) {
	bestVal := fit[0]
	bestIdx := 0
	for i := 1; i < len(fit); i++ {
		if math.IsNaN(fit[i]) || math.IsInf(fit[i], 0) {
			fit[i] = 0.0
		}
		if fit[i] > bestVal {
			bestVal = fit[i]
			bestIdx = i
		}
	}
	return bestVal, bestIdx
}

func argmax(arr []float64) int {
	bestI := 0
	bestV := arr[0]
	for i := 1; i < len(arr); i++ {
		if math.IsNaN(arr[i]) || math.IsInf(arr[i], 0) {
			continue
		}
		if arr[i] > bestV {
			bestV = arr[i]
			bestI = i
		}
	}
	return bestI
}

func visualizeResults(bp *phase.Phase, trainX, trainY, testX, testY *mat.Dense) {
	fmt.Println("\n--- Visualizing Results ---")
	trainRows, _ := trainX.Dims()
	fmt.Printf("Train (first 5):\n")
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
		fmt.Printf("  Sample %d => pred=%d, actual=%d, raw=%.2f\n", i, pred, actual, vals[pred])
	}

	testRows, _ := testX.Dims()
	fmt.Printf("\nTest (first 5):\n")
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
		fmt.Printf("  Sample %d => pred=%d, actual=%d, raw=%.2f\n", i, pred, actual, vals[pred])
	}

	trainExact, trainClose, trainProx := evaluateAccuracy(bp, trainX, trainY)
	testExact, testClose, testProx := evaluateAccuracy(bp, testX, testY)
	fmt.Printf("\nFinal Train Exact Accuracy: %.2f%%\n", trainExact*100)
	fmt.Printf("Final Train Close Accuracy (%.0f%% threshold): %.2f%%\n", closeThreshold*100, trainClose*100)
	fmt.Printf("Final Train Proximity Score: %.2f%%\n", trainProx)
	fmt.Printf("Final Test  Exact Accuracy: %.2f%%\n", testExact*100)
	fmt.Printf("Final Test  Close Accuracy (%.0f%% threshold): %.2f%%\n", closeThreshold*100, testClose*100)
	fmt.Printf("Final Test  Proximity Score: %.2f%%\n\n", testProx)
}

// ============================================================================
// Save/Load
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
	fmt.Println("Auto clamp triggered due to Inf/NaN neuron values. Clamping all to range:", minVal, maxVal)

	numWorkers := runtime.NumCPU()
	neuronIDs := make([]int, 0, len(bp.Neurons))
	for id := range bp.Neurons {
		neuronIDs = append(neuronIDs, id)
	}
	chunkSize := (len(neuronIDs) + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(neuronIDs) {
			end = len(neuronIDs)
		}
		if start >= len(neuronIDs) {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				id := neuronIDs[i]
				neuron := bp.Neurons[id]
				if math.IsNaN(neuron.Value) || math.IsInf(neuron.Value, 0) {
					neuron.Value = 0
				}
				if neuron.Value > maxVal {
					neuron.Value = maxVal
				} else if neuron.Value < minVal {
					neuron.Value = minVal
				}
				if math.IsNaN(neuron.CellState) || math.IsInf(neuron.CellState, 0) {
					neuron.CellState = 0
				}
				if neuron.CellState > maxVal {
					neuron.CellState = maxVal
				} else if neuron.CellState < minVal {
					neuron.CellState = minVal
				}
				if math.IsNaN(neuron.Bias) || math.IsInf(neuron.Bias, 0) {
					neuron.Bias = 0
				}
				if neuron.Bias > maxVal {
					neuron.Bias = maxVal
				} else if neuron.Bias < minVal {
					neuron.Bias = minVal
				}
				for j := range neuron.Connections {
					w := neuron.Connections[j][1]
					if math.IsNaN(w) || math.IsInf(w, 0) {
						w = 0
					}
					if w > maxVal {
						w = maxVal
					} else if w < minVal {
						w = minVal
					}
					neuron.Connections[j][1] = w
				}
			}
		}(start, end)
	}
	wg.Wait()
}
