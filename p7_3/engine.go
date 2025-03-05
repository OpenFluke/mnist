package main

import (
	"encoding/binary"
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
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"
	epsilon  = 0.01 // Tolerance for floating-point comparisons
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

func main() {
	// Seed the random number generator.
	rand.Seed(time.Now().UnixNano())

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

	// Use 80% of the data for training/checkpointing (48,000 samples)
	trainSize := int(0.8 * float64(len(trainInputs)))
	trainSamples := make([]Sample, trainSize)
	for i := 0; i < trainSize; i++ {
		trainSamples[i] = Sample{Inputs: trainInputs[i], Label: trainLabels[i]}
	}
	fmt.Printf("Using %d samples for training and checkpointing\n", len(trainSamples))

	// Step 3: Create the initial neural network.
	fmt.Println("Step 3: Creating the initial neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	currentExactAcc, currentClosenessBins, currentApproxScore := evaluateModelWithCheckpoints(bp, trainSamples)
	fmt.Printf("Initial model metrics: ExactAcc=%.4f, ClosenessBins=%v, ApproxScore=%.4f\n",
		currentExactAcc, formatClosenessBins(currentClosenessBins), currentApproxScore)

	// Step 4: Create initial checkpoint with all training data (48,000 samples).
	fmt.Println("Step 4: Creating initial checkpoint with all training data...")
	checkpoints := bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
	fmt.Printf("Checkpoint created with %d samples\n", len(checkpoints))

	// Set up multi-threading to use 80% of CPU cores.
	numCPUs := int(float64(runtime.NumCPU()) * 0.8)
	runtime.GOMAXPROCS(numCPUs)
	fmt.Printf("Using %d CPU cores (80%% of %d)\n", numCPUs, runtime.NumCPU())

	// Generational loop
	generation := 1
	for generation <= 10 { // Terminate after 10 generations
		fmt.Printf("\n=== Generation %d ===\n", generation)

		// Step 5: Create and evaluate 10 copies in parallel.
		results := make(chan ModelResult, 10)
		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(copyID int) {
				defer wg.Done()
				result := evolveModel(bp, trainSamples, checkpoints, copyID)
				results <- result
			}(i)
		}

		// Wait for all copies to finish and collect results.
		go func() {
			wg.Wait()
			close(results)
		}()

		// Find the best model from the copies based on any metric improving.
		bestExactAcc := currentExactAcc
		bestClosenessBins := currentClosenessBins
		bestApproxScore := currentApproxScore
		var bestBP *phase.Phase
		for result := range results {
			fmt.Printf("Copy %d added %d neurons, ExactAcc=%.4f, ClosenessBins=%v, ApproxScore=%.4f\n",
				result.BP.ID, result.NeuronsAdded, result.ExactAcc, formatClosenessBins(result.ClosenessBins), result.ApproxScore)
			if isImproved(result.ExactAcc, result.ClosenessBins, result.ApproxScore, bestExactAcc, bestClosenessBins, bestApproxScore) {
				bestExactAcc = result.ExactAcc
				bestClosenessBins = result.ClosenessBins
				bestApproxScore = result.ApproxScore
				bestBP = result.BP
			}
		}

		// Step 6: Update model if any metric improved.
		if bestBP != nil {
			fmt.Printf("Improved model found! New metrics: ExactAcc=%.4f, ClosenessBins=%v, ApproxScore=%.4f (old: %.4f, %v, %.4f)\n",
				bestExactAcc, formatClosenessBins(bestClosenessBins), bestApproxScore, currentExactAcc, formatClosenessBins(currentClosenessBins), currentApproxScore)
			bp = bestBP
			currentExactAcc = bestExactAcc
			currentClosenessBins = bestClosenessBins
			currentApproxScore = bestApproxScore
			// Recreate checkpoint with the new model, covering all training data.
			fmt.Println("Recreating checkpoint with updated model...")
			checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
			fmt.Printf("New checkpoint created with %d samples\n", len(checkpoints))
		} else {
			fmt.Println("No significant improvement in any metric found in this generation.")
		}

		generation++
	}

	fmt.Println("\nDone. Training completed.")
}

// evolveModel creates a copy of the model and attempts to improve it by adding neurons.
func evolveModel(originalBP *phase.Phase, samples []Sample, checkpoints []map[int]map[string]interface{}, copyID int) ModelResult {
	// Copy the original model using the existing Copy method.
	bp := originalBP.Copy()
	bp.ID = copyID // Assign the copyID as the new ID

	// Baseline metrics using checkpoint evaluation
	bestExactAcc, bestClosenessBins, bestApproxScore := evaluateModelWithCheckpoints(bp, samples, checkpoints)
	neuronsAdded := 0
	maxAttempts := 5 // Number of attempts to add neurons

	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Add 1-50 neurons randomly with 1-50 connections each
		numToAdd := rand.Intn(50) + 1 // 1 to 50 neurons
		for i := 0; i < numToAdd; i++ {
			newN := bp.AddNeuronFromPreOutputs("dense", "", 1, 50) // Random activation, 1-50 connections
			if newN == nil {
				continue
			}
			bp.AddNewNeuronToOutput(newN.ID)
			neuronsAdded++
		}

		// Evaluate the new model using checkpoint-based metrics
		newExactAcc, newClosenessBins, newApproxScore := evaluateModelWithCheckpoints(bp, samples, checkpoints)
		if isImproved(newExactAcc, newClosenessBins, newApproxScore, bestExactAcc, bestClosenessBins, bestApproxScore) {
			bestExactAcc = newExactAcc
			bestClosenessBins = newClosenessBins
			bestApproxScore = newApproxScore
			// Keep this configuration and continue adding
		} else {
			// Revert to the original copy and try again
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

// evaluateModelWithCheckpoints evaluates the model using checkpoints and returns exactAcc, closenessBins, and approxScore.
func evaluateModelWithCheckpoints(bp *phase.Phase, samples []Sample, checkpoints ...[]map[int]map[string]interface{}) (float64, []float64, float64) {
	// If checkpoints are provided, use them; otherwise, compute new ones
	var chkpts []map[int]map[string]interface{}
	if len(checkpoints) > 0 && len(checkpoints[0]) == len(samples) {
		chkpts = checkpoints[0]
	} else {
		chkpts = bp.CheckpointAllHiddenNeurons(getInputs(samples), 1)
	}

	// Convert labels to float64 for EvaluateMetricsFromCheckpoints
	labels := make([]float64, len(samples))
	for i, sample := range samples {
		labels[i] = float64(sample.Label)
	}

	// Evaluate using the checkpoint-based method
	exactAcc, closenessBins, approxScore := bp.EvaluateMetricsFromCheckpoints(chkpts, labels)
	return exactAcc, closenessBins, approxScore
}

// isImproved checks if any metric (exactAcc, closenessBins, approxScore) has improved.
func isImproved(newExactAcc float64, newClosenessBins []float64, newApproxScore float64, oldExactAcc float64, oldClosenessBins []float64, oldApproxScore float64) bool {
	// Check ExactAcc improvement
	if newExactAcc > oldExactAcc+epsilon {
		return true
	}

	// Check ApproxScore improvement
	if newApproxScore > oldApproxScore+epsilon {
		return true
	}

	// Check ClosenessBins improvement (higher bins increase, e.g., shift from >90% to 80-90%)
	newClosenessScore := weightedClosenessScore(newClosenessBins)
	oldClosenessScore := weightedClosenessScore(oldClosenessBins)
	if newClosenessScore > oldClosenessScore+epsilon {
		return true
	}

	return false
}

// weightedClosenessScore computes a score for closenessBins, weighting higher bins (lower difference) more.
func weightedClosenessScore(bins []float64) float64 {
	score := 0.0
	for i, bin := range bins {
		weight := float64(10 - i) // 10 down to 1
		score += bin * weight
	}
	return score / 100.0 // Normalize to [0, 100] range
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

// getInputs extracts inputs from a slice of samples.
func getInputs(samples []Sample) []map[int]float64 {
	inputs := make([]map[int]float64, len(samples))
	for i, sample := range samples {
		inputs[i] = sample.Inputs
	}
	return inputs
}

// Helper Functions (unchanged)
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

func saveModelOrWarn(bp *phase.Phase, filePath string) {
	if err := bp.SaveToJSON(filePath); err != nil {
		fmt.Printf("Warning: failed to save model to %s => %v\n", filePath, err)
	} else {
		fmt.Printf("Model saved to %s\n", filePath)
	}
}
