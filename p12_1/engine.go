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
	maxConnections          = 600
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
	//selectedModel = initModel() // Assign the returned model
	//generation()

	evaluateOnTestSetFullForward("models/gen_35.json")
}

func evaluateOnTestSetFullForward(modelPath string) {
	// Load the model from the specified path
	data, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model file %s: %v", modelPath, err)
	}
	loadedModel := phase.NewPhase()
	if err := json.Unmarshal(data, loadedModel); err != nil {
		log.Fatalf("Failed to deserialize model from %s: %v", modelPath, err)
	}
	fmt.Printf("Loaded model from %s\n", modelPath)

	// Ensure test samples are loaded
	if len(testSamples) == 0 {
		fmt.Println("Test samples not loaded yet. Loading MNIST data...")
		setupMnist() // Load testSamples if not already loaded
	}

	// Prepare to collect metrics
	var exactMatches int
	binCounts := make([]int, 10) // 10 bins for ClosenessBins (0.0 to 1.0 in steps of 0.1)
	var sumApprox float64

	// Assume output nodes correspond to classes 0-9
	outputNodeStart := loadedModel.OutputNodes[0]

	for _, sample := range testSamples {
		// Run full forward pass
		loadedModel.Forward(sample.Inputs, 1) // 1 timestep, as in training
		outputs := loadedModel.GetOutputs()

		// Find predicted class (highest output value)
		predClass := -1
		maxVal := -math.MaxFloat64
		for id, val := range outputs {
			if val > maxVal {
				maxVal = val
				predClass = id - outputNodeStart // Map to class 0-9
			}
		}

		trueClass := sample.Label

		// ExactAcc: percentage of correct predictions
		if predClass == trueClass {
			exactMatches++
		}

		// ClosenessBins: how close the correct class's output is to 1.0
		correctOutputID := outputNodeStart + trueClass
		correctVal, exists := outputs[correctOutputID]
		if !exists {
			log.Fatalf("Output for correct class %d not found in sample", trueClass)
		}
		difference := math.Abs(correctVal - 1.0)
		if difference > 1 {
			difference = 1 // Cap at 1.0
		}
		binIndex := int(difference * 10) // Bins: 0 (<0.1), 1 (0.1-0.2), ..., 9 (>0.9)
		if binIndex > 9 {
			binIndex = 9
		}
		binCounts[binIndex]++

		// ApproxScore: confidence in the correct class
		sumApprox += correctVal
	}

	// Compute final metrics
	nSamples := len(testSamples)
	exactAcc := (float64(exactMatches) / float64(nSamples)) * 100.0
	closenessBins := make([]float64, 10)
	for i := 0; i < 10; i++ {
		closenessBins[i] = (float64(binCounts[i]) / float64(nSamples)) * 100.0
	}
	approxScore := (sumApprox / float64(nSamples)) * 100.0 // Scaled to 100
	closenessQuality := loadedModel.ComputeClosenessQuality(closenessBins)

	// Print metrics
	fmt.Println("Test Set Metrics (Full Forward Pass):")
	fmt.Printf("  ExactAcc: %.4f%%\n", exactAcc)
	fmt.Printf("  ClosenessBins: %v\n", phase.FormatClosenessBins(closenessBins))
	fmt.Printf("  ApproxScore: %.4f\n", approxScore)
	fmt.Printf("  ClosenessQuality: %.4f\n", closenessQuality)
}

func setupMnist() {

	numCPUs = runtime.NumCPU()
	numWorkers = int(float64(numCPUs) * 0.8)
	runtime.GOMAXPROCS(numWorkers)
	fmt.Printf("Using %d workers (80%% of %d CPUs)\n", numWorkers, numCPUs)

	// Load MNIST dataset
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

	// Set trainSize to 80% of the total data
	trainSize = int(0.8 * float64(len(trainInputs)))

	// Calculate testSize as the remaining 20%
	testSize = len(trainInputs) - trainSize

	// Initialize trainSamples and testSamples slices
	trainSamples = make([]phase.Sample, trainSize)
	testSamples = make([]phase.Sample, testSize)

	// Populate trainSamples (first 80%)
	for i := 0; i < trainSize; i++ {
		trainSamples[i] = phase.Sample{Inputs: trainInputs[i], Label: trainLabels[i]}
	}

	// Populate testSamples (remaining 20%)
	for i := 0; i < testSize; i++ {
		testSamples[i] = phase.Sample{Inputs: trainInputs[trainSize+i], Label: trainLabels[trainSize+i]}
	}

	fmt.Printf("Using %d samples for training and %d samples for testing\n", len(trainSamples), len(testSamples))
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
	trainLabels := phase.GetLabels(&trainSamples)

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
		labels := phase.GetLabels(&trainSamples)
		currentExactAcc, currentClosenessBins, currentApproxScore = baseBP.EvaluateWithCheckpoints(checkpointFolder, &initialCheckpoint, labels)
		currentClosenessQuality = baseBP.ComputeClosenessQuality(currentClosenessBins)
	}

	var bestSelected phase.ModelResult
	bestImprovement := -math.MaxFloat64
	for i := 0; i < numTournaments; i++ {
		candidate := baseBP.TournamentSelection(results, currentExactAcc, currentClosenessQuality, currentApproxScore, 3)
		candidateImprovement := baseBP.ComputeTotalImprovement(candidate, currentExactAcc, currentClosenessQuality, currentApproxScore)
		if candidateImprovement > bestImprovement {
			bestImprovement = candidateImprovement
			bestSelected = candidate
		}
	}

	if bestImprovement > 0 {
		newClosenessQuality := baseBP.ComputeClosenessQuality(bestSelected.ClosenessBins)
		deltaExactAcc := bestSelected.ExactAcc - currentExactAcc
		deltaApproxScore := bestSelected.ApproxScore - currentApproxScore
		deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

		fmt.Printf("Improved model found in generation %d with total improvement %.4f via tournament selection\n", generation, bestImprovement)
		fmt.Printf("Metric improvements:\n")
		fmt.Printf("  ExactAcc: %.4f → %.4f (Δ %.4f)\n", currentExactAcc, bestSelected.ExactAcc, deltaExactAcc)
		fmt.Printf("  ClosenessBins: %v → %v\n", phase.FormatClosenessBins(currentClosenessBins), phase.FormatClosenessBins(bestSelected.ClosenessBins))
		fmt.Printf("  ClosenessQuality: %.4f → %.4f (Δ %.4f)\n", currentClosenessQuality, newClosenessQuality, deltaClosenessQuality)
		fmt.Printf("  ApproxScore: %.4f → %.4f (Δ %.4f)\n", currentApproxScore, bestSelected.ApproxScore, deltaApproxScore)

		selectedModel = bestSelected.BP
		currentExactAcc = bestSelected.ExactAcc
		currentClosenessBins = bestSelected.ClosenessBins
		currentApproxScore = bestSelected.ApproxScore

		modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", currentGenNumber))
		if err := selectedModel.SaveToJSON(modelPath); err != nil {
			log.Printf("Failed to save model for generation %d: %v", generation, err)
		}
		return true // Indicate improvement
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
