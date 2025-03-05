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
	"sync"
	"time"

	"phase" // Replace with your actual import path
)

const (
	baseURL                = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir               = "mnist_data"
	epsilon                = 0.01 // Tolerance for floating-point comparisons
	modelDir               = "models"
	maxGenerations         = 500
	initialNumModels       = 10  // Starting number of models per generation
	maxNumModels           = 100 // Maximum number of models per generation
	noImprovementThreshold = 5   // Generations without improvement before increasing models
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
	rand.Seed(time.Now().UnixNano())
	processStartTime := time.Now()
	fmt.Printf("Process started at %s\n", processStartTime.Format("2006-01-02 15:04:05"))

	if err := os.MkdirAll(modelDir, 0755); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}

	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	fmt.Println("Step 2: Loading MNIST training dataset...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(trainInputs))

	trainSize := int(0.8 * float64(len(trainInputs)))
	trainSamples := make([]Sample, trainSize)
	for i := 0; i < trainSize; i++ {
		trainSamples[i] = Sample{Inputs: trainInputs[i], Label: trainLabels[i]}
	}
	fmt.Printf("Using %d samples for training and checkpointing\n", len(trainSamples))

	fmt.Println("Step 3: Creating the initial neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	currentExactAcc, currentClosenessBins, currentApproxScore := evaluateModelWithCheckpoints(bp, trainSamples)
	currentClosenessQuality := computeClosenessQuality(currentClosenessBins)
	fmt.Printf("Initial model metrics:\n")
	fmt.Printf("  ExactAcc: %.4f\n", currentExactAcc)
	fmt.Printf("  ClosenessBins: %v\n", formatClosenessBins(currentClosenessBins))
	fmt.Printf("  ApproxScore: %.4f\n", currentApproxScore)
	fmt.Printf("  ClosenessQuality: %.4f\n", currentClosenessQuality)

	if err := saveModel(bp, filepath.Join(modelDir, "gen_0.json")); err != nil {
		log.Printf("Failed to save initial model: %v", err)
	}

	fmt.Println("Step 4: Creating initial checkpoint with all training data...")
	checkpoints := bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
	fmt.Printf("Checkpoint created with %d samples\n", len(checkpoints))

	numCPUs := runtime.NumCPU()
	numWorkers := int(float64(numCPUs) * 0.8)
	runtime.GOMAXPROCS(numWorkers)
	fmt.Printf("Using %d workers (80%% of %d CPUs)\n", numWorkers, numCPUs)

	currentNumModels := initialNumModels
	generationsWithoutImprovement := 0

	for generation := 1; generation <= maxGenerations; generation++ {
		genStartTime := time.Now()
		fmt.Printf("\n=== Generation %d started at %s with %d models ===\n", generation, genStartTime.Format("2006-01-02 15:04:05"), currentNumModels)

		currentClosenessQuality = computeClosenessQuality(currentClosenessBins)

		jobChan := make(chan int, currentNumModels)
		resultChan := make(chan ModelResult, currentNumModels)

		for i := 0; i < numWorkers; i++ {
			go func() {
				for job := range jobChan {
					resultChan <- evolveModel(bp, trainSamples, checkpoints, job)
				}
			}()
		}

		for i := 0; i < currentNumModels; i++ {
			jobChan <- i
		}
		close(jobChan)

		results := make([]ModelResult, 0, currentNumModels)
		for i := 0; i < currentNumModels; i++ {
			results = append(results, <-resultChan)
		}

		bestTotalImprovement := -math.MaxFloat64
		var bestResult ModelResult
		for _, result := range results {
			newClosenessQuality := computeClosenessQuality(result.ClosenessBins)
			deltaExactAcc := result.ExactAcc - currentExactAcc
			deltaApproxScore := result.ApproxScore - currentApproxScore
			deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

			// Normalize improvements
			normDeltaExactAcc := deltaExactAcc / 100.0                 // Max ExactAcc = 100
			normDeltaApproxScore := deltaApproxScore / 100.0           // Max ApproxScore = 100
			normDeltaClosenessQuality := deltaClosenessQuality / 100.0 // Max ClosenessQuality = 100

			// Weighted sum
			weightExactAcc := 0.3
			weightCloseness := 0.4
			weightApproxScore := 0.3
			totalImprovement := (weightExactAcc * normDeltaExactAcc) +
				(weightCloseness * normDeltaClosenessQuality) +
				(weightApproxScore * normDeltaApproxScore)

			if totalImprovement > bestTotalImprovement {
				bestTotalImprovement = totalImprovement
				bestResult = result
			}
		}

		if bestTotalImprovement > 0 {
			newClosenessQuality := computeClosenessQuality(bestResult.ClosenessBins)
			deltaExactAcc := bestResult.ExactAcc - currentExactAcc
			deltaApproxScore := bestResult.ApproxScore - currentApproxScore
			deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

			fmt.Printf("Improved model found in generation %d with total improvement %.4f\n", generation, bestTotalImprovement)
			fmt.Printf("Metric improvements:\n")
			fmt.Printf("  ExactAcc: %.4f → %.4f (Δ %.4f)\n", currentExactAcc, bestResult.ExactAcc, deltaExactAcc)
			fmt.Printf("  ClosenessBins: %v → %v\n", formatClosenessBins(currentClosenessBins), formatClosenessBins(bestResult.ClosenessBins))
			fmt.Printf("  ClosenessQuality: %.4f → %.4f (Δ %.4f)\n", currentClosenessQuality, newClosenessQuality, deltaClosenessQuality)
			fmt.Printf("  ApproxScore: %.4f → %.4f (Δ %.4f)\n", currentApproxScore, bestResult.ApproxScore, deltaApproxScore)

			bp = bestResult.BP
			currentExactAcc = bestResult.ExactAcc
			currentClosenessBins = bestResult.ClosenessBins
			currentApproxScore = bestResult.ApproxScore

			modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", generation))
			if err := saveModel(bp, modelPath); err != nil {
				log.Printf("Failed to save model for generation %d: %v", generation, err)
			}

			fmt.Println("Recreating checkpoint with updated model...")
			checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
			fmt.Printf("New checkpoint created with %d samples\n", len(checkpoints))

			generationsWithoutImprovement = 0
			currentNumModels = initialNumModels
		} else {
			fmt.Println("No significant improvement found in this generation.")
			generationsWithoutImprovement++
			if generationsWithoutImprovement >= noImprovementThreshold && currentNumModels < maxNumModels {
				currentNumModels += 10
				if currentNumModels > maxNumModels {
					currentNumModels = maxNumModels
				}
				fmt.Printf("Increasing number of models to %d for next generation.\n", currentNumModels)
			}
		}

		genEndTime := time.Now()
		genDuration := genEndTime.Sub(genStartTime).Seconds()
		fmt.Printf("Generation %d finished at %s, duration: %.2f seconds\n", generation, genEndTime.Format("2006-01-02 15:04:05"), genDuration)
	}

	processEndTime := time.Now()
	totalProcessTime := processEndTime.Sub(processStartTime).Seconds()
	fmt.Printf("\nProcess finished at %s, total time: %.2f seconds\n", processEndTime.Format("2006-01-02 15:04:05"), totalProcessTime)
}

func evolveModel(originalBP *phase.Phase, samples []Sample, checkpoints []map[int]map[string]interface{}, copyID int) ModelResult {
	// Start with a sandbox copy of the current best model.
	bestBP := originalBP.Copy()
	bestBP.ID = copyID

	bestExactAcc, bestClosenessBins, bestApproxScore := evaluateModelWithCheckpoints(bestBP, samples, checkpoints)
	neuronsAdded := 0

	// Use a loop that continues as long as improvements are found.
	// Also use a safety limit to avoid infinite loops.
	maxIterations := 1000
	iterations := 0
	improvementFound := true

	for improvementFound && iterations < maxIterations {
		improvementFound = false
		iterations++

		// Work on a fresh copy of the current best model.
		currentBP := bestBP.Copy()

		// Add a small, incremental number of neurons.
		// (Adjust the range as needed.)
		numToAdd := rand.Intn(5) + 1 // e.g. add between 1 and 5 neurons at a time.
		for i := 0; i < numToAdd; i++ {
			newNeuron := currentBP.AddNeuronFromPreOutputs("dense", "", 1, 50)
			if newNeuron != nil {
				currentBP.AddNewNeuronToOutput(newNeuron.ID)
				neuronsAdded++
			}
		}

		// Evaluate the updated model using the same checkpoint.
		newExactAcc, newClosenessBins, newApproxScore := evaluateModelWithCheckpoints(currentBP, samples, checkpoints)
		newClosenessQuality := computeClosenessQuality(newClosenessBins)
		bestClosenessQuality := computeClosenessQuality(bestClosenessBins)

		// If any metric improves, update the sandbox model.
		if newExactAcc > bestExactAcc+epsilon ||
			newClosenessQuality > bestClosenessQuality+epsilon ||
			newApproxScore > bestApproxScore+epsilon {
			bestBP = currentBP
			bestExactAcc = newExactAcc
			bestClosenessBins = newClosenessBins
			bestApproxScore = newApproxScore
			improvementFound = true
		}
	}

	return ModelResult{
		BP:            bestBP,
		ExactAcc:      bestExactAcc,
		ClosenessBins: bestClosenessBins,
		ApproxScore:   bestApproxScore,
		NeuronsAdded:  neuronsAdded,
	}
}

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

	return bp.EvaluateMetricsFromCheckpoints(chkpts, labels)
}

func computeClosenessQuality(bins []float64) float64 {
	// Sum the percentage of samples in bins >= 50% closeness (difference <= 0.5)
	quality := 0.0
	for i := 5; i < len(bins); i++ { // Bins 50-60%, 60-70%, ..., >90%
		quality += bins[i]
	}
	return quality
}

func getInputs(samples []Sample) []map[int]float64 {
	inputs := make([]map[int]float64, len(samples))
	for i, sample := range samples {
		inputs[i] = sample.Inputs
	}
	return inputs
}

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
