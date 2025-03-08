package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"phase" // Replace with your actual import path, e.g., "github.com/yourusername/phase"
)

const (
	baseURL           = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir          = "mnist_data"
	modelFile         = "saved_model.json"
	checkpointFile    = "saved_checkpoint.json"
	modelDir          = "models"
	checkpointFolder  = ""
	currentNumModels  = 30
	useMultithreading = true
	epsilon           = 0.00
	numTournaments    = 5
	evalWithMultiCore = false
)

var (
	numCPUs                 int
	numWorkers              int
	initialCheckpoint       []map[int]map[string]interface{}
	trainSize               int
	testSize                int
	trainSamples            []phase.Sample
	testSamples             []phase.Sample
	trainInputs             []map[int]float64
	trainLabels             []int
	processStartTime        time.Time
	avgGenTime              time.Duration
	startGeneration         = 1
	maxGenerations          = 500
	currentGenNumber        = 1
	layers                  = []int{784, 1, 10}
	hiddenAct               = "relu"
	outputAct               = "linear"
	selectedModel           *phase.Phase
	maxIterations           = 10
	maxConsecutiveFailures  = 5
	minConnections          = 10
	maxConnections          = 6000
	currentExactAcc         float64
	currentClosenessBins    []float64
	currentApproxScore      float64
	currentClosenessQuality float64
	minNeuronsToAdd         = 1
	maxNeuronsToAdd         = 1
)

func main() {
	rand.Seed(time.Now().UnixNano())
	processStartTime = time.Now()
	fmt.Printf("Process started at %s\n", processStartTime.Format("2006-01-02 15:04:05"))
	setupMnist()

	selectedModel = initModel() // Assign the returned model
	generation()
}

func setupMnist() {
	numCPUs = runtime.NumCPU()
	numWorkers = int(float64(numCPUs) * 0.8)
	runtime.GOMAXPROCS(numWorkers)
	fmt.Printf("Using %d workers (80%% of %d CPUs)\n", numWorkers, numCPUs)

	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	fmt.Println("Step 2: Loading MNIST training dataset...")
	var err error
	trainInputs, trainLabels, err = loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(trainInputs))

	trainSize = int(0.8 * float64(len(trainInputs)))
	testSize = len(trainInputs) - trainSize

	// Initialize a temporary model to get OutputNodes
	tempModel := phase.NewPhaseWithLayers(layers, hiddenAct, outputAct)
	outputNodes := tempModel.OutputNodes

	// Populate trainSamples with ExpectedOutputs
	trainSamples = make([]phase.Sample, trainSize)
	for i := 0; i < trainSize; i++ {
		expectedOutputs := createExpectedOutputs(trainLabels[i], outputNodes)
		trainSamples[i] = phase.Sample{
			Inputs:          trainInputs[i],
			ExpectedOutputs: expectedOutputs,
		}
	}

	// Populate testSamples with ExpectedOutputs
	testSamples = make([]phase.Sample, testSize)
	for i := 0; i < testSize; i++ {
		expectedOutputs := createExpectedOutputs(trainLabels[trainSize+i], outputNodes)
		testSamples[i] = phase.Sample{
			Inputs:          trainInputs[trainSize+i],
			ExpectedOutputs: expectedOutputs,
		}
	}

	fmt.Printf("Using %d samples for training and %d samples for testing\n", len(trainSamples), len(testSamples))
}

// createExpectedOutputs converts an integer label to a one-hot encoded map for the output neurons.
func createExpectedOutputs(label int, outputNodes []int) map[int]float64 {
	expected := make(map[int]float64)
	for i, nodeID := range outputNodes {
		if i == label {
			expected[nodeID] = 1.0
		} else {
			expected[nodeID] = 0.0
		}
	}
	return expected
}

func initModel() *phase.Phase {
	// Check for the latest generation model in the models directory
	files, err := os.ReadDir(modelDir)
	if err != nil {
		if os.IsNotExist(err) {
			// If the directory doesn’t exist, create it and proceed with a new model
			if err := os.MkdirAll(modelDir, 0755); err != nil {
				log.Fatalf("Failed to create models directory: %v", err)
			}
		} else {
			log.Fatalf("Failed to read models directory: %v", err)
		}
	}

	latestGen := -1
	var latestFile string
	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "gen_") && strings.HasSuffix(file.Name(), ".json") {
			genStr := strings.TrimPrefix(strings.TrimSuffix(file.Name(), ".json"), "gen_")
			gen, err := strconv.Atoi(genStr)
			if err == nil && gen > latestGen {
				latestGen = gen
				latestFile = file.Name()
			}
		}
	}

	// If a latest generation model exists, load it
	if latestGen != -1 {
		modelPath := filepath.Join(modelDir, latestFile)
		fmt.Println("Saved model found. Loading it...")
		data, err := os.ReadFile(modelPath)
		if err != nil {
			log.Fatalf("Failed to read saved model %s: %v", modelPath, err)
		}
		loadedBP := phase.NewPhase()
		if err := json.Unmarshal(data, loadedBP); err != nil { // Use Unmarshal directly since POC used DeserializesFromJSON
			log.Fatalf("Failed to deserialize model from %s: %v", modelPath, err)
		}
		fmt.Printf("Loaded model from %s (generation %d)\n", modelPath, latestGen)
		return loadedBP
	}

	// Otherwise, create a new model
	fmt.Println("No saved model found. Creating new neural network...")
	return phase.NewPhaseWithLayers(layers, hiddenAct, outputAct)
}

func testModelPerformance(checkpoint []map[int]map[string]interface{}) {
	if selectedModel == nil {
		log.Fatalf("Cannot test model performance: selectedModel is nil")
	}

	// Test on training set (using provided checkpoint)
	fmt.Println("Testing current model performance on training set...")
	trainLabels := phase.GetLabels(&trainSamples, selectedModel.OutputNodes) // Fixed: Added second argument

	trainExactAcc, trainClosenessBins, trainApproxScore := selectedModel.EvaluateWithCheckpoints(checkpointFolder, &checkpoint, trainLabels)
	trainClosenessQuality := selectedModel.ComputeClosenessQuality(trainClosenessBins)

	fmt.Printf("Training Set Metrics (Checkpoint size: %d, Labels size: %d):\n", len(checkpoint), len(*trainLabels))
	fmt.Printf("  ExactAcc: %.4f%%\n", trainExactAcc)
	fmt.Printf("  ClosenessBins: %v\n", phase.FormatClosenessBins(trainClosenessBins))
	fmt.Printf("  ApproxScore: %.4f\n", trainApproxScore)
	fmt.Printf("  ClosenessQuality: %.4f\n", trainClosenessQuality)
}

func createInitialCheckpoint() {
	if selectedModel == nil {
		log.Fatalf("Cannot create initial checkpoint: selectedModel is nil")
	}
	if len(trainSamples) == 0 {
		log.Fatalf("Cannot create initial checkpoint: trainSamples is empty")
	}
	fmt.Println("Creating initial checkpoint for training samples...")

	if evalWithMultiCore {
		initialCheckpoint = selectedModel.CheckpointPreOutputNeuronsMultiCore(checkpointFolder, getInputs(trainSamples), 1)
	} else {
		initialCheckpoint = selectedModel.CheckpointPreOutputNeurons(checkpointFolder, getInputs(trainSamples), 1)
	}

	fmt.Printf("Initial checkpoint created with %d samples\n", len(initialCheckpoint))
}

// Helper function (since it’s not in your code yet)
func getInputs(samples []phase.Sample) []map[int]float64 {
	inputs := make([]map[int]float64, len(samples))
	for i, sample := range samples {
		inputs[i] = sample.Inputs
	}
	return inputs
}

func generation() {
	var totalGenTime time.Duration // To accumulate total time across generations
	genCount := 0                  // To count the number of generations
	createInitialCheckpoint()

	testModelPerformance(initialCheckpoint)

	for generation := startGeneration; generation <= maxGenerations; generation++ {
		genStartTime := time.Now()
		fmt.Printf("\n=== GEN %d started %s\n", generation, genStartTime.Format("2006-01-02 15:04:05"))

		// Simulate some work here (if any) - currently empty in your code
		improved := training() // Check if training improved the model

		if improved {
			fmt.Println("Regenerating checkpoint due to model improvement...")
			createInitialCheckpoint() // Update checkpoint with the improved selectedModel
		}

		// Calculate generation duration and update total time
		genDuration := time.Since(genStartTime)
		totalGenTime += genDuration
		genCount++

		// Update average generation time
		avgGenTime = totalGenTime / time.Duration(genCount)

		// Calculate full running time since process start
		fullRunningTime := time.Since(processStartTime)

		currentGenNumber++
		fmt.Printf("=== GEN %d finished. Gen time: %s, Full: %s, Avg: %s\n",
			generation, genDuration, fullRunningTime, avgGenTime)
		// Evaluate the model on the test set after each generation
	}
}

func training() bool {
	results := make([]phase.ModelResult, currentNumModels) // Pre-allocate with exact size
	baseBP := selectedModel                                // Use the global initialized model

	if useMultithreading {
		numWorkers := int(float64(runtime.NumCPU()) * 0.8) // 80% of CPU cores
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, numWorkers) // Limit concurrency to 80% of CPUs

		for i := 0; i < currentNumModels; i++ {
			wg.Add(1)
			semaphore <- struct{}{} // Acquire semaphore slot
			go func(workerID int) {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore slot
				result := baseBP.Grow(minNeuronsToAdd, maxNeuronsToAdd, evalWithMultiCore, checkpointFolder, baseBP, &trainSamples, &initialCheckpoint, workerID, maxIterations, maxConsecutiveFailures, minConnections, maxConnections, epsilon)
				results[workerID] = result
			}(i)
		}

		wg.Wait()
	} else {
		for i := 0; i < currentNumModels; i++ {
			result := baseBP.Grow(minNeuronsToAdd, maxNeuronsToAdd, evalWithMultiCore, checkpointFolder, baseBP, &trainSamples, &initialCheckpoint, i, maxIterations, maxConsecutiveFailures, minConnections, maxConnections, epsilon)
			results[i] = result
		}
	}

	// Initialize metrics if not set (first generation)
	if currentExactAcc == 0 && len(currentClosenessBins) == 0 && currentApproxScore == 0 {
		labels := phase.GetLabels(&trainSamples, baseBP.OutputNodes) // Updated
		currentExactAcc, currentClosenessBins, currentApproxScore = baseBP.EvaluateWithCheckpoints(checkpointFolder, &initialCheckpoint, labels)
		currentClosenessQuality = baseBP.ComputeClosenessQuality(currentClosenessBins)
	}

	/*var bestSelected phase.ModelResult
	bestImprovement := -math.MaxFloat64
	for i := 0; i < numTournaments; i++ {
		candidate := baseBP.TournamentSelection(results, currentExactAcc, currentClosenessQuality, currentApproxScore, 3)
		candidateImprovement := baseBP.ComputeTotalImprovement(candidate, currentExactAcc, currentClosenessQuality, currentApproxScore)
		if candidateImprovement > bestImprovement {
			bestImprovement = candidateImprovement
			bestSelected = candidate
		}
	}*/
	bestSelected, bestImprovement := baseBP.SelectBestModel(results, currentExactAcc, currentClosenessQuality, currentApproxScore)

	if bestImprovement > 0 {
		newClosenessQuality := baseBP.ComputeClosenessQuality(bestSelected.ClosenessBins)
		deltaExactAcc := bestSelected.ExactAcc - currentExactAcc
		deltaApproxScore := bestSelected.ApproxScore - currentApproxScore
		deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

		fmt.Printf("Improved model found in generation %d with total improvement %.4f \n", currentGenNumber, bestImprovement)
		fmt.Printf("Metric improvements:\n")
		fmt.Printf("  ExactAcc: %.4f → %.4f (Δ %.4f)\n", currentExactAcc, bestSelected.ExactAcc, deltaExactAcc)
		fmt.Printf("  ClosenessBins: %v → %v\n", "\n"+phase.FormatClosenessBins(currentClosenessBins)+"\n", phase.FormatClosenessBins(bestSelected.ClosenessBins))
		fmt.Printf("  ClosenessQuality: %.4f → %.4f (Δ %.4f)\n", currentClosenessQuality, newClosenessQuality, deltaClosenessQuality)
		fmt.Printf("  ApproxScore: %.4f → %.4f (Δ %.4f)\n", currentApproxScore, bestSelected.ApproxScore, deltaApproxScore)

		// Update globals
		selectedModel = bestSelected.BP
		currentExactAcc = bestSelected.ExactAcc
		currentClosenessBins = bestSelected.ClosenessBins
		currentApproxScore = bestSelected.ApproxScore
		currentClosenessQuality = newClosenessQuality // Explicitly update this

		modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", currentGenNumber))
		if err := selectedModel.SaveToJSON(modelPath); err != nil {
			log.Printf("Failed to save model for generation %d: %v", currentGenNumber, err)
		}
		return true
	}
	return false // No improvement
}

// --- Helper Functions for Data Loading and Downloads ---

func ensureMNISTDownloads(bp *phase.Phase, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []string{"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"}
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

func loadMNIST(dir, imageFile, labelFile string, limit int) ([]map[int]float64, []int, error) {
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

	inputs := make([]map[int]float64, limit)
	labels := make([]int, limit)
	buf := make([]byte, 784) // 28x28
	for i := 0; i < limit; i++ {
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data at sample %d: %w", i, err)
		}
		inputMap := make(map[int]float64, 784)
		for px := 0; px < 784; px++ {
			inputMap[px] = float64(buf[px]) / 255.0
		}
		inputs[i] = inputMap

		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data at sample %d: %w", i, err)
		}
		labels[i] = int(lblByte[0])
	}
	return inputs, labels, nil
}
