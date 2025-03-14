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

	epochTrain()
	//generation()
}

func epochTrain() {
	// Print when training starts
	fmt.Println("Starting training with gradient descent...")

	// Loop over 100 epochs
	for epoch := 0; epoch < 100; epoch++ {
		// Record the start time of the epoch
		startTime := time.Now()

		// Print when the epoch begins
		fmt.Printf("Epoch %d started\n", epoch+1)

		createInitialCheckpoint()

		testModelPerformance(initialCheckpoint)
		// Loop over all training samples
		for i, sample := range trainSamples {
			// Remap inputs to match actual input neuron IDs
			inputIDs := selectedModel.InputNodes
			if len(inputIDs) != 784 {
				log.Fatalf("Expected 784 input neurons, but got %d", len(inputIDs))
			}
			remappedInputs := make(map[int]float64)
			for px := 0; px < 784; px++ {
				remappedInputs[inputIDs[px]] = sample.Inputs[px]
			}

			// Create expected outputs using actual output neuron IDs
			outputIDs := selectedModel.OutputNodes
			if len(outputIDs) != 10 {
				log.Fatalf("Expected 10 output neurons, but got %d", len(outputIDs))
			}
			expectedOutputs := make(map[int]float64)
			for j := 0; j < 10; j++ {
				if j == sample.Label {
					expectedOutputs[outputIDs[j]] = 1.0
				} else {
					expectedOutputs[outputIDs[j]] = 0.0
				}
			}

			// Train the network
			selectedModel.TrainNetwork(remappedInputs, expectedOutputs, 0.01, -5.0, 5.0)

			// Print progress every 1000 samples to avoid flooding the console
			if i%1000 == 0 {
				fmt.Printf("Epoch %d: Processing sample %d of %d, label: %d, first input: %.2f\n",
					epoch+1, i, len(trainSamples), sample.Label, sample.Inputs[0])
				break
			}
		}

		createInitialCheckpoint()

		testModelPerformance(initialCheckpoint)

		// Calculate and print the time taken for the epoch
		duration := time.Since(startTime)
		fmt.Printf("Epoch %d completed in %.2f seconds\n", epoch+1, duration.Seconds())

	}

	// Print when training is fully done
	fmt.Println("Training completed successfully!")

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
