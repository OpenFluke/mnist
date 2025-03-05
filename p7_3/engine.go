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
	"sync"
	"time"

	"phase" // Replace with your actual import path, e.g., "github.com/you/phase"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	epsilon   = 0.01 // Tolerance for floating-point comparisons
	modelDir  = "models"
	numModels = 100 // Number of models to process per generation
)

type Sample struct {
	Inputs map[int]float64
	Label  int
}

type ModelResult struct {
	BP            *phase.Phase
	ExactAcc      float64   // Exact accuracy in [0, 100]
	ClosenessBins []float64 // Closeness bins in [0, 100] per bin
	ApproxScore   float64   // Approx score in [0, 100]
	NeuronsAdded  int
}

var (
	improvedMutex sync.Mutex
	improved      bool
)

func main() {
	// Seed the random number generator.
	rand.Seed(time.Now().UnixNano())

	// Record process start time.
	processStartTime := time.Now()
	fmt.Printf("Process started at %s\n", processStartTime.Format("2006-01-02 15:04:05"))

	// Create models directory.
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}

	// Step 1: Ensure MNIST is downloaded.
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Step 2: Load MNIST training data.
	fmt.Println("Step 2: Loading MNIST training dataset...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(trainInputs))

	// Use 80% of the data for training/checkpointing (48,000 samples).
	trainSize := int(0.8 * float64(len(trainInputs)))
	trainSamples := make([]Sample, trainSize)
	for i := 0; i < trainSize; i++ {
		trainSamples[i] = Sample{Inputs: trainInputs[i], Label: trainLabels[i]}
	}
	fmt.Printf("Using %d samples for training and checkpointing\n", len(trainSamples))

	// Step 3: Initialize the neural network.
	fmt.Println("Step 3: Creating the initial neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	currentExactAcc, currentClosenessBins, currentApproxScore := evaluateModelWithCheckpoints(bp, trainSamples)
	fmt.Printf("Initial model metrics:\n")
	fmt.Printf("  ExactAcc: %.4f\n", currentExactAcc)
	fmt.Printf("  ClosenessBins: %v\n", formatClosenessBins(currentClosenessBins))
	fmt.Printf("  ApproxScore: %.4f\n", currentApproxScore)

	// Save initial model as gen_0.
	if err := saveModel(bp, filepath.Join(modelDir, "gen_0.json")); err != nil {
		log.Printf("Failed to save initial model: %v", err)
	}

	// Step 4: Create initial checkpoint.
	fmt.Println("Step 4: Creating initial checkpoint with all training data...")
	checkpoints := bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
	fmt.Printf("Checkpoint created with %d samples\n", len(checkpoints))

	// Calculate number of workers (80% of CPU cores).
	numCPUs := runtime.NumCPU()
	numWorkers := int(float64(numCPUs) * 0.8)
	runtime.GOMAXPROCS(numWorkers)
	fmt.Printf("Using %d workers (80%% of %d CPUs)\n", numWorkers, numCPUs)

	// Generational loop (10 generations).
	for generation := 1; generation <= 10; generation++ {
		// Reset improvement flag.
		improved = false

		// Record generation start time.
		genStartTime := time.Now()
		fmt.Printf("\n=== Generation %d started at %s ===\n", generation, genStartTime.Format("2006-01-02 15:04:05"))

		// Set up worker pool.
		jobChan := make(chan int, numModels)
		resultChan := make(chan ModelResult, numModels)

		// Start worker goroutines.
		for i := 0; i < numWorkers; i++ {
			go func() {
				for job := range jobChan {
					if improved {
						return
					}
					result := evolveModel(bp, trainSamples, checkpoints, job)
					resultChan <- result
				}
			}()
		}

		// Send jobs to workers.
		for i := 0; i < numModels; i++ {
			jobChan <- i
		}
		close(jobChan)

		// Collect results from workers.
		resultsCollected := 0
		for resultsCollected < numModels {
			result := <-resultChan
			resultsCollected++
			improvedMutex.Lock()
			if result.ExactAcc > currentExactAcc+epsilon && !improved {
				improved = true
				bp = result.BP
				currentExactAcc = result.ExactAcc
				currentClosenessBins = result.ClosenessBins
				currentApproxScore = result.ApproxScore
				// Save the improved model.
				modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", generation))
				if err := saveModel(bp, modelPath); err != nil {
					log.Printf("Failed to save model for generation %d: %v", generation, err)
				}
				improvedMutex.Unlock()
				break // Stop collecting results and move to next generation.
			}
			improvedMutex.Unlock()
		}

		// Update if a better model is found.
		if improved {
			fmt.Printf("Improved model found in generation %d!\n", generation)
			fmt.Printf("Metric improvements:\n")
			fmt.Printf("  ExactAcc: %.4f → %.4f\n", currentExactAcc-epsilon, currentExactAcc)
			fmt.Printf("  ApproxScore: %.4f → %.4f\n", currentApproxScore, currentApproxScore)
			fmt.Printf("  ClosenessBins: %v\n", formatClosenessBins(currentClosenessBins))
			// Recreate checkpoints for the new model.
			fmt.Println("Recreating checkpoint with updated model...")
			checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
			fmt.Printf("New checkpoint created with %d samples\n", len(checkpoints))
		} else {
			fmt.Println("No significant improvement in ExactAcc found in this generation.")
		}

		// Record generation end time and calculate duration.
		genEndTime := time.Now()
		genDuration := genEndTime.Sub(genStartTime).Seconds()
		fmt.Printf("Generation %d finished at %s, duration: %.2f seconds\n", generation, genEndTime.Format("2006-01-02 15:04:05"), genDuration)
	}

	// Record and display process end time and total time.
	processEndTime := time.Now()
	totalProcessTime := processEndTime.Sub(processStartTime).Seconds()
	fmt.Printf("\nProcess finished at %s, total time: %.2f seconds\n", processEndTime.Format("2006-01-02 15:04:05"), totalProcessTime)
}

// evolveModel creates a model copy and improves it by adding neurons.
func evolveModel(originalBP *phase.Phase, samples []Sample, checkpoints []map[int]map[string]interface{}, copyID int) ModelResult {
	bp := originalBP.Copy()
	bp.ID = copyID

	bestExactAcc, bestClosenessBins, bestApproxScore := evaluateModelWithCheckpoints(bp, samples, checkpoints)
	neuronsAdded := 0
	maxAttempts := 5

	for attempt := 0; attempt < maxAttempts; attempt++ {
		improvedMutex.Lock()
		if improved {
			improvedMutex.Unlock()
			return ModelResult{BP: bp, ExactAcc: bestExactAcc, ClosenessBins: bestClosenessBins, ApproxScore: bestApproxScore, NeuronsAdded: neuronsAdded}
		}
		improvedMutex.Unlock()

		numToAdd := rand.Intn(50) + 1
		for i := 0; i < numToAdd; i++ {
			newN := bp.AddNeuronFromPreOutputs("dense", "", 1, 50)
			if newN == nil {
				continue
			}
			bp.AddNewNeuronToOutput(newN.ID)
			neuronsAdded++
		}

		newExactAcc, newClosenessBins, newApproxScore := evaluateModelWithCheckpoints(bp, samples, checkpoints)
		if newExactAcc > bestExactAcc+epsilon {
			bestExactAcc = newExactAcc
			bestClosenessBins = newClosenessBins
			bestApproxScore = newApproxScore
		} else {
			bp = originalBP.Copy()
			bp.ID = copyID
			neuronsAdded = 0
		}
	}

	return ModelResult{
		BP:            bp,
		ExactAcc:      bestExactAcc,
		ClosenessBins: bestClosenessBins,
		ApproxScore:   bestApproxScore,
		NeuronsAdded:  neuronsAdded,
	}
}

// evaluateModelWithCheckpoints evaluates a model using checkpoints.
func evaluateModelWithCheckpoints(bp *phase.Phase, samples []Sample, checkpoints ...[]map[int]map[string]interface{}) (float64, []float64, float64) {
	var chkpts []map[int]map[string]interface{}
	if len(checkpoints) > 0 && len(checkpoints[0]) == len(samples) {
		chkpts = checkpoints[0]
	} else {
		chkpts = bp.CheckpointAllHiddenNeurons(getInputs(samples), 1)
	}

	labels := make([]float64, len(samples))
	for i, sample := range samples {
		labels[i] = float64(sample.Label)
	}

	exactAcc, closenessBins, approxScore := bp.EvaluateMetricsFromCheckpoints(chkpts, labels)
	return exactAcc, closenessBins, approxScore
}

// getInputs extracts inputs from samples.
func getInputs(samples []Sample) []map[int]float64 {
	inputs := make([]map[int]float64, len(samples))
	for i, sample := range samples {
		inputs[i] = sample.Inputs
	}
	return inputs
}

// saveModel saves the model to a JSON file.
func saveModel(bp *phase.Phase, filePath string) error {
	data, err := json.Marshal(bp)
	if err != nil {
		return fmt.Errorf("failed to serialize model: %v", err)
	}
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write model to file: %v", err)
	}
	fmt.Printf("Model saved to %s\n", filePath)
	return nil
}

// loadModel loads a model from a JSON file.
func loadModel(filePath string) (*phase.Phase, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %v", err)
	}
	bp := phase.NewPhase()
	if err := json.Unmarshal(data, bp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %v", err)
	}
	fmt.Printf("Model loaded from %s\n", filePath)
	return bp, nil
}

// ensureMNISTDownloads downloads and unzips MNIST data if needed.
func ensureMNISTDownloads(bp *phase.Phase, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
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

// loadMNIST loads MNIST images and labels from files.
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
	buf := make([]byte, 784)
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

// formatClosenessBins formats the closeness bins for readable output.
func formatClosenessBins(bins []float64) string {
	s := "["
	for i, bin := range bins {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%.2f", bin)
	}
	s += "]"
	return s
}
