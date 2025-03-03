package main

import (
	"encoding/binary"
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

	"phase" // Adjust to your local import path
)

const (
	baseURL                        = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir                       = "mnist_data"
	batchSize                      = 50     // Adjustable number of networks per generation
	samplePercentage               = 5      // Adjustable sample percentage (1% = 1, 10% = 10)
	maxNoImprovementForConnections = 10     // Increase connections after this many generations without improvement
	maxNoImprovementForNeurons     = 20     // Increase neurons per attempt after this many without improvement
	testDir                        = "test" // Directory to save models
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Step 1: Download MNIST dataset if not already present
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Step 2: Load MNIST training and testing datasets
	fmt.Println("Step 2: Loading MNIST training and testing datasets...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	testInputs, testLabels, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading testing MNIST: %v", err)
	}

	// Step 3: Split training data into 80% training and 20% validation
	fmt.Println("Step 3: Splitting training data into 80% training and 20% validation...")
	totalTrainSamples := 60000
	trainSplit := int(0.8 * float64(totalTrainSamples)) // 48,000 training samples
	trainData := trainInputs[:trainSplit]
	trainLbls := trainLabels[:trainSplit]
	valData := trainInputs[trainSplit:totalTrainSamples]
	valLbls := trainLabels[trainSplit:totalTrainSamples]

	_ = trainLbls
	_ = testLabels

	// Print dataset sizes
	fmt.Printf("Training set:   %d samples\n", len(trainData))
	fmt.Printf("Validation set: %d samples\n", len(valData))
	fmt.Printf("Testing set:    %d samples\n", len(testInputs))

	// Step 4: Load latest model or create a new one
	fmt.Println("Step 4: Checking for latest saved model...")
	bp, latestGen, err := loadLatestModel(bp)
	if err != nil {
		log.Printf("No valid saved model found, starting fresh: %v", err)
		fmt.Println("Step 4: Creating a dummy neural network...")
		bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
		latestGen = 0
	} else {
		fmt.Printf("Loaded model from generation %d\n", latestGen)
	}

	// Convert validation labels to float64
	valLabelsFloat := make([]float64, len(valLbls))
	for i, lbl := range valLbls {
		valLabelsFloat[i] = float64(lbl)
	}

	// Step 5: Initial evaluation
	fmt.Println("Step 5: Initial evaluation of the network...")
	exactAcc, closeAccs, proximity := bp.EvaluateMetrics(valData, valLabelsFloat)

	// Print initial evaluation results
	fmt.Printf("\nValidation Set Metrics (Initial Network at Gen %d):\n", latestGen)
	fmt.Printf("Exact Accuracy:  %f%%\n", exactAcc)
	fmt.Printf("Proximity Score: %f\n", proximity)
	fmt.Println("Close Accuracies at different thresholds:")
	for i, acc := range closeAccs {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: %f%%\n", threshold, acc)
	}

	// Step 6: Run 500 generation cycles starting from latestGen
	fmt.Println("\nStep 6: Starting 500 generation cycles of training with batch size", batchSize, "and sample percentage", samplePercentage, "% from generation", latestGen+1, "...")
	bestBP := runGenerationCycles(bp, valData, valLabelsFloat, 500, batchSize, 5, 1, 10, latestGen)

	// Step 7: Final evaluation with the best model
	fmt.Println("\nStep 7: Final evaluation with the best model after 500 generations...")
	exactAcc, closeAccs, proximity = bestBP.EvaluateMetrics(valData, valLabelsFloat)
	fmt.Printf("\nValidation Set Metrics (Best Network After 500 Generations):\n")
	fmt.Printf("Exact Accuracy:  %f%%\n", exactAcc)
	fmt.Printf("Proximity Score: %f\n", proximity)
	fmt.Println("Close Accuracies at different thresholds:")
	for i, acc := range closeAccs {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: %f%%\n", threshold, acc)
	}

	fmt.Println("\nTraining and evaluation complete.")
}

// loadLatestModel checks the test folder for the latest model and loads it
func loadLatestModel(defaultBP *phase.Phase) (*phase.Phase, int, error) {
	if err := os.MkdirAll(testDir, os.ModePerm); err != nil {
		return defaultBP, 0, fmt.Errorf("failed to create test directory: %v", err)
	}

	files, err := os.ReadDir(testDir)
	if err != nil {
		return defaultBP, 0, fmt.Errorf("failed to read test directory: %v", err)
	}

	latestGen := -1
	var latestFile string
	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "model_gen") && strings.HasSuffix(file.Name(), ".json") {
			genStr := strings.TrimPrefix(strings.TrimSuffix(file.Name(), ".json"), "model_gen")
			gen, err := strconv.Atoi(genStr)
			if err == nil && gen > latestGen {
				latestGen = gen
				latestFile = file.Name()
			}
		}
	}

	if latestGen == -1 {
		return defaultBP, 0, nil // No models found
	}

	data, err := os.ReadFile(filepath.Join(testDir, latestFile))
	if err != nil {
		return defaultBP, 0, fmt.Errorf("failed to read model file %s: %v", latestFile, err)
	}

	bp := phase.NewPhase()
	if err := bp.DeserializesFromJSON(string(data)); err != nil {
		return defaultBP, 0, fmt.Errorf("failed to deserialize model from %s: %v", latestFile, err)
	}

	return bp, latestGen, nil
}

// runGenerationCycles runs the specified number of generation cycles
func runGenerationCycles(bp *phase.Phase, valData []map[int]float64, valLabels []float64, generations, batchSize, maxAttempts, minConnectionsInitial, maxConnectionsInitial, startGen int) *phase.Phase {
	currentBest := bp
	_, _, bestApprox := currentBest.EvaluateMetrics(valData, valLabels)
	noImprovementCount := 0
	minConnections := minConnectionsInitial
	maxConnections := maxConnectionsInitial
	neuronsPerAttempt := 1

	for gen := startGen + 1; gen <= generations; gen++ {
		fmt.Printf("\n=== Generation %d/%d ===\n", gen, generations)

		// Adjust complexity based on no improvement
		if noImprovementCount >= maxNoImprovementForNeurons {
			neuronsPerAttempt++
			noImprovementCount = 0 // Reset after increasing neurons
			fmt.Printf("Increasing complexity: Now adding %d neurons per attempt\n", neuronsPerAttempt)
		} else if noImprovementCount >= maxNoImprovementForConnections {
			minConnections++
			maxConnections += 2 // Increase range more significantly
			fmt.Printf("Increasing complexity: Connection range now %d-%d\n", minConnections, maxConnections)
		}

		// Select samplePercentage of validation data
		sampleSize := int(float64(samplePercentage) / 100 * float64(len(valData))) // e.g., 1% of 12,000 = 120, 10% = 1200
		sampleIndices := rand.Perm(len(valData))[:sampleSize]
		sampleInputs := make([]map[int]float64, sampleSize)
		sampleLabels := make([]float64, sampleSize)
		for i, idx := range sampleIndices {
			sampleInputs[i] = valData[idx]
			sampleLabels[i] = valLabels[idx]
		}

		// Train a batch and get an improved model (if any)
		improvedBP := trainWithNeuronBatches(currentBest, valData, valLabels, sampleInputs, sampleLabels, batchSize, maxAttempts, minConnections, maxConnections, neuronsPerAttempt)

		if improvedBP != nil {
			_, _, newApprox := improvedBP.EvaluateMetrics(valData, valLabels)
			if newApprox > bestApprox {
				currentBest = improvedBP
				bestApprox = newApprox
				noImprovementCount = 0 // Reset on improvement
				fmt.Printf("Generation %d: Found improvement\n", gen)
				fmt.Printf("Current Prox: %f\n", bestApprox)
				fmt.Printf("New Prox:     %f\n", newApprox)

				// Save the improved model
				filename := filepath.Join(testDir, fmt.Sprintf("model_gen%d.json", gen))
				if err := currentBest.SaveToJSON(filename); err != nil {
					log.Printf("Failed to save model for generation %d: %v", gen, err)
				} else {
					fmt.Printf("Saved improved model to %s\n", filename)
				}
			} else {
				fmt.Printf("Generation %d: Improved model found but not better than current best (%f vs %f)\n", gen, newApprox, bestApprox)
				noImprovementCount++
			}
		} else {
			fmt.Printf("Generation %d: Tried %d possible changes - No improvement found\n", gen, batchSize*maxAttempts)
			noImprovementCount++
		}

		// Print current best metrics every 50 generations
		if gen%50 == 0 {
			exactAcc, closeAccs, proximity := currentBest.EvaluateMetrics(valData, valLabels)
			fmt.Printf("\nProgress at Generation %d:\n", gen)
			fmt.Printf("Exact Accuracy:  %f%%\n", exactAcc)
			fmt.Printf("Proximity Score: %f\n", proximity)
			fmt.Println("Close Accuracies at different thresholds:")
			for i, acc := range closeAccs {
				threshold := (i + 1) * 10
				fmt.Printf("  %d%% Threshold: %f%%\n", threshold, acc)
			}
		}
	}

	return currentBest
}

// trainWithNeuronBatches implements a single generation's training methodology
func trainWithNeuronBatches(bp *phase.Phase, valData []map[int]float64, valLabels []float64, sampleInputs []map[int]float64, sampleLabels []float64, batchSize, maxAttempts, minConnections, maxConnections, neuronsPerAttempt int) *phase.Phase {
	// Step 1: Create batch of neural networks
	batch := make([]*phase.Phase, batchSize)
	for i := 0; i < batchSize; i++ {
		batch[i] = bp.Copy()
	}

	// Step 2: Initial metrics for the original network on the sample subset
	_, _, baseApprox := bp.EvaluateMetrics(sampleInputs, sampleLabels)
	fmt.Printf("\nTraining: Base Prox on %d%% Sample (%d samples) - %f\n", samplePercentage, len(sampleInputs), baseApprox)

	// Step 3: Train networks in parallel
	numCores := int(float64(runtime.NumCPU()) * 0.8) // Use 80% of CPU cores
	if numCores < 1 {
		numCores = 1
	}
	results := make(chan struct {
		Index int
		Net   *phase.Phase
		Prox  float64
		Found bool
	}, batchSize)
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, numCores)

	for i, net := range batch {
		wg.Add(1)
		go func(idx int, network *phase.Phase) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			fmt.Printf("Training network %d/%d on %d%% sample (Neurons: %d)...\n", idx+1, batchSize, samplePercentage, len(network.Neurons))
			bestNet := network.Copy()
			bestProx := baseApprox
			foundImprovement := false

			for attempt := 1; attempt <= maxAttempts; attempt++ {
				// Work on a fresh copy for each attempt
				currentNet := network.Copy()

				// Add 'neuronsPerAttempt' number of neurons
				for n := 0; n < neuronsPerAttempt; n++ {
					currentNet.AddRandomNeuron("", "", minConnections, maxConnections)
					newNeuronID := currentNet.GetNextNeuronID() - 1
					currentNet.RewireOutputsThroughNewNeuron(newNeuronID)
				}

				// Evaluate on the sample subset
				_, _, newApprox := currentNet.EvaluateMetrics(sampleInputs, sampleLabels)
				fmt.Printf("Attempt %d: New Prox - %f (Neurons: %d)\n", attempt, newApprox, len(currentNet.Neurons))

				// Check for prox improvement
				if newApprox > baseApprox {
					fmt.Printf("Found prox improvement in network %d:\n", idx+1)
					fmt.Printf("Current Prox: %f\n", baseApprox)
					fmt.Printf("New Prox:     %f\n", newApprox)
					bestNet = currentNet.Copy()
					bestProx = newApprox
					foundImprovement = true
					break // Stop if prox is improved
				}

				// Track best approx
				if newApprox > bestProx {
					bestProx = newApprox
					bestNet = currentNet.Copy()
				}
			}

			results <- struct {
				Index int
				Net   *phase.Phase
				Prox  float64
				Found bool
			}{idx, bestNet, bestProx, foundImprovement}
		}(i, net)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect training results
	improvedNetworks := make([]*phase.Phase, 0, batchSize)
	bestApproxNet := bp
	bestApproxScore := baseApprox

	for result := range results {
		if result.Found {
			improvedNetworks = append(improvedNetworks, result.Net)
		}
		if result.Prox > bestApproxScore {
			bestApproxScore = result.Prox
			bestApproxNet = result.Net
		}
	}

	// Step 4: Select improved networks
	if len(improvedNetworks) == 0 && bestApproxScore > baseApprox {
		// Fallback to best approx improvement
		fmt.Printf("No direct improvement found, falling back to best approx in network:\n")
		fmt.Printf("Current Prox: %f\n", baseApprox)
		fmt.Printf("New Prox:     %f\n", bestApproxScore)
		improvedNetworks = append(improvedNetworks, bestApproxNet)
	} else if len(improvedNetworks) == 0 {
		fmt.Println("No networks improved on the sample.")
		return nil
	}

	// Step 5: Evaluate improved networks on full dataset in parallel
	numCores = int(float64(runtime.NumCPU()) * 0.8) // Recompute in case CPU load changed
	if numCores < 1 {
		numCores = 1
	}
	fullResults := make(chan struct {
		Index int
		Net   *phase.Phase
		Score float64
	}, len(improvedNetworks))
	var fullWG sync.WaitGroup
	semaphore = make(chan struct{}, numCores) // Reinitialize semaphore

	for i, net := range improvedNetworks {
		fullWG.Add(1)
		go func(idx int, n *phase.Phase) {
			defer fullWG.Done()
			semaphore <- struct{}{}        // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			_, _, approx := n.EvaluateMetrics(valData, valLabels)
			fullResults <- struct {
				Index int
				Net   *phase.Phase
				Score float64
			}{idx, n, approx}
		}(i, net)
	}

	go func() {
		fullWG.Wait()
		close(fullResults)
	}()

	// Collect full dataset evaluation results
	improvedResults := make([]struct {
		Net   *phase.Phase
		Score float64
	}, len(improvedNetworks))
	for result := range fullResults {
		improvedResults[result.Index] = struct {
			Net   *phase.Phase
			Score float64
		}{result.Net, result.Score}
	}

	// Step 6: Compare with original network's full dataset score
	_, _, originalApprox := bp.EvaluateMetrics(valData, valLabels)
	bestNet := bp
	bestScore := originalApprox
	improved := false

	for _, res := range improvedResults {
		fmt.Printf("Evaluating improved network: Prox Score = %f (Original = %f)\n", res.Score, originalApprox)
		if res.Score > originalApprox {
			if res.Score > bestScore {
				bestScore = res.Score
				bestNet = res.Net
				improved = true
			}
		}
	}

	if improved {
		fmt.Printf("Improved network found with Prox Score %f (Original %f)\n", bestScore, originalApprox)
		return bestNet
	}
	fmt.Println("No overall improvement found on full dataset.")
	return nil
}

// Helper Functions (unchanged)
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
