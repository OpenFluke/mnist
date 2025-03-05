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

	"phase" // Replace with your actual import path
)

const (
	baseURL                = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir               = "mnist_data"
	epsilon                = 0.001 // Tolerance for floating-point comparisons
	modelDir               = "models"
	maxGenerations         = 500
	initialNumModels       = 10  // Starting number of models per generation
	maxNumModels           = 100 // Maximum number of models per generation
	noImprovementThreshold = 5   // Generations without improvement before increasing models
	maxIterations          = 10
	useMultithreading      = true
	checkpointBatchSize    = 10                    // Number of checkpoints to load per batch
	baseCheckpointDir      = "cpcache"             // Base directory for all checkpoints
	sharedCheckpointSubDir = "current_checkpoints" // Subdirectory for the best model's checkpoints
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
	improvedMutex      sync.Mutex
	improved           bool
	useFileCheckpoints bool = true // Toggle for file-based checkpointing (default: true)
)

// **computeTotalImprovement** calculates the weighted sum of improvements for a model.
func computeTotalImprovement(result ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64) float64 {
	newClosenessQuality := computeClosenessQuality(result.ClosenessBins)
	deltaExactAcc := result.ExactAcc - currentExactAcc
	deltaApproxScore := result.ApproxScore - currentApproxScore
	deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

	// Normalize improvements
	normDeltaExactAcc := deltaExactAcc / 100.0
	normDeltaApproxScore := deltaApproxScore / 100.0
	normDeltaClosenessQuality := deltaClosenessQuality / 100.0

	// Weighted sum
	weightExactAcc := 0.3
	weightCloseness := 0.4
	weightApproxScore := 0.3
	return (weightExactAcc * normDeltaExactAcc) +
		(weightCloseness * normDeltaClosenessQuality) +
		(weightApproxScore * normDeltaApproxScore)
}

// **tournamentSelection** selects the best model from a random subset of results.
func tournamentSelection(results []ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64, tournamentSize int) ModelResult {
	if len(results) < tournamentSize {
		tournamentSize = len(results)
	}
	perm := rand.Perm(len(results))
	selectedIndices := perm[:tournamentSize]
	bestIdx := selectedIndices[0]
	bestImprovement := computeTotalImprovement(results[bestIdx], currentExactAcc, currentClosenessQuality, currentApproxScore)
	for _, idx := range selectedIndices[1:] {
		improvement := computeTotalImprovement(results[idx], currentExactAcc, currentClosenessQuality, currentApproxScore)
		if improvement > bestImprovement {
			bestImprovement = improvement
			bestIdx = idx
		}
	}
	return results[bestIdx]
}

// **findLatestGeneration** finds the highest generation number from saved models.
func findLatestGeneration() (int, bool) {
	files, err := os.ReadDir(modelDir)
	if err != nil {
		log.Fatalf("Failed to read models directory: %v", err)
	}
	maxGen := -1
	for _, file := range files {
		if strings.HasPrefix(file.Name(), "gen_") && strings.HasSuffix(file.Name(), ".json") {
			genStr := strings.TrimSuffix(strings.TrimPrefix(file.Name(), "gen_"), ".json")
			gen, err := strconv.Atoi(genStr)
			if err == nil && gen > maxGen {
				maxGen = gen
			}
		}
	}
	if maxGen == -1 {
		return 0, false
	}
	return maxGen, true
}

// **loadModel** loads a model from a JSON file.
func loadModel(filePath string) (*phase.Phase, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %v", err)
	}
	var bp phase.Phase
	if err := json.Unmarshal(data, &bp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %v", err)
	}
	bp.InitializeActivationFunctions() // Initialize activation map
	return &bp, nil
}

func main() {
	rand.Seed(time.Now().UnixNano())
	processStartTime := time.Now()
	fmt.Printf("Process started at %s\n", processStartTime.Format("2006-01-02 15:04:05"))

	// Ensure directories exist
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}
	if useFileCheckpoints {
		if err := os.MkdirAll(baseCheckpointDir, 0755); err != nil {
			log.Fatalf("Failed to create base checkpoint directory %s: %v", baseCheckpointDir, err)
		}
		// Clean up old temporary checkpoint directories, but keep sharedCheckpointSubDir
		entries, err := os.ReadDir(baseCheckpointDir)
		if err == nil {
			for _, entry := range entries {
				if entry.IsDir() && entry.Name() != sharedCheckpointSubDir {
					tempDir := filepath.Join(baseCheckpointDir, entry.Name())
					if err := os.RemoveAll(tempDir); err != nil {
						log.Printf("Failed to remove old temporary directory %s: %v", tempDir, err)
					}
				}
			}
		}
	}

	checkpointDir := filepath.Join(baseCheckpointDir, sharedCheckpointSubDir)

	// Load MNIST dataset
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

	// Variables to hold the model and its metrics
	var currentExactAcc float64
	var currentClosenessBins []float64
	var currentApproxScore float64
	var currentClosenessQuality float64
	var checkpoints []map[int]map[string]interface{}
	startGeneration := 1

	// Check for existing generation models
	latestGen, hasGenModels := findLatestGeneration()
	if hasGenModels {
		fmt.Printf("Loading latest generation model (gen_%d.json)...\n", latestGen)
		modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", latestGen))
		bp, err = loadModel(modelPath)
		if err != nil {
			log.Fatalf("Failed to load model from %s: %v", modelPath, err)
		}
		startGeneration = latestGen + 1

		// Check if checkpoints are up-to-date
		latestModelFile := filepath.Join(baseCheckpointDir, "latest_model.txt")
		data, err := os.ReadFile(latestModelFile)
		if err == nil && string(data) == fmt.Sprintf("%d", latestGen) {
			// Checkpoints are up-to-date, proceed with loading metrics
			fmt.Printf("Using existing checkpoints in %s\n", checkpointDir)
		} else {
			// Recompute checkpoints
			fmt.Println("Checkpoints missing or outdated, recomputing...")
			if useFileCheckpoints {
				if err := os.RemoveAll(checkpointDir); err != nil && !os.IsNotExist(err) {
					log.Printf("Failed to clean up checkpoint directory %s: %v", checkpointDir, err)
				}
				if err := bp.SaveCheckpointsToDirectory(getInputs(trainSamples), 1, checkpointDir); err != nil {
					log.Fatalf("Failed to save checkpoints for loaded model: %v", err)
				}
				fmt.Printf("Checkpoints saved to %s\n", checkpointDir)
				// Update latest_model.txt
				if err := os.WriteFile(latestModelFile, []byte(fmt.Sprintf("%d", latestGen)), 0644); err != nil {
					log.Printf("Failed to write latest_model.txt: %v", err)
				}
			} else {
				checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
				fmt.Printf("Checkpoints recomputed with %d samples in memory\n", len(checkpoints))
			}
		}
	} else {
		// No generation models, check for initial_best.json
		initialModelPath := filepath.Join(modelDir, "initial_best.json")
		if _, err := os.Stat(initialModelPath); err == nil {
			fmt.Println("Loading initial best model (initial_best.json)...")
			bp, err = loadModel(initialModelPath)
			if err != nil {
				log.Fatalf("Failed to load initial model from %s: %v", initialModelPath, err)
			}
			startGeneration = 1

			// Check if checkpoints are up-to-date
			latestModelFile := filepath.Join(baseCheckpointDir, "latest_model.txt")
			data, err := os.ReadFile(latestModelFile)
			if err == nil && string(data) == "initial" {
				fmt.Printf("Using existing checkpoints in %s\n", checkpointDir)
			} else {
				fmt.Println("Checkpoints missing or outdated, recomputing...")
				if useFileCheckpoints {
					if err := os.RemoveAll(checkpointDir); err != nil && !os.IsNotExist(err) {
						log.Printf("Failed to clean up checkpoint directory %s: %v", checkpointDir, err)
					}
					if err := bp.SaveCheckpointsToDirectory(getInputs(trainSamples), 1, checkpointDir); err != nil {
						log.Fatalf("Failed to save checkpoints for initial model: %v", err)
					}
					fmt.Printf("Checkpoints saved to %s\n", checkpointDir)
					// Update latest_model.txt
					if err := os.WriteFile(latestModelFile, []byte("initial"), 0644); err != nil {
						log.Printf("Failed to write latest_model.txt: %v", err)
					}
				} else {
					checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
					fmt.Printf("Checkpoints recomputed with %d samples in memory\n", len(checkpoints))
				}
			}
		} else {
			// No models found, initialize new ones
			fmt.Println("No saved models found. Step 3: Selecting the best initial neural network from multiple candidates...")
			numCPUs := runtime.NumCPU()
			numWorkers := int(float64(numCPUs) * 0.8)
			runtime.GOMAXPROCS(numWorkers)
			fmt.Printf("Using %d workers (80%% of %d CPUs)\n", numWorkers, numCPUs)

			layers := []int{784, 64, 10}
			hiddenAct := "relu"
			outputAct := "linear"
			bp, currentExactAcc, currentClosenessBins, currentApproxScore = selectBestInitialModel(initialNumModels, layers, hiddenAct, outputAct, trainSamples, numWorkers)
			currentClosenessQuality = computeClosenessQuality(currentClosenessBins)
			fmt.Println("Selected best initial model with metrics:")
			fmt.Printf("  ExactAcc: %.4f\n", currentExactAcc)
			fmt.Printf("  ClosenessBins: %v\n", formatClosenessBins(currentClosenessBins))
			fmt.Printf("  ApproxScore: %.4f\n", currentApproxScore)
			fmt.Printf("  ClosenessQuality: %.4f\n", currentClosenessQuality)

			if err := saveModel(bp, initialModelPath); err != nil {
				log.Printf("Failed to save best initial model: %v", err)
			}

			fmt.Println("Step 4: Creating initial checkpoint with all training data...")
			if useFileCheckpoints {
				if err := os.RemoveAll(checkpointDir); err != nil && !os.IsNotExist(err) {
					log.Printf("Failed to clean up checkpoint directory %s: %v", checkpointDir, err)
				}
				if err := bp.SaveCheckpointsToDirectory(getInputs(trainSamples), 1, checkpointDir); err != nil {
					log.Fatalf("Failed to save initial checkpoints: %v", err)
				}
				fmt.Printf("Initial checkpoints saved to %s\n", checkpointDir)
				// Update latest_model.txt
				latestModelFile := filepath.Join(baseCheckpointDir, "latest_model.txt")
				if err := os.WriteFile(latestModelFile, []byte("initial"), 0644); err != nil {
					log.Printf("Failed to write latest_model.txt: %v", err)
				}
			} else {
				checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
				fmt.Printf("Checkpoint created with %d samples in memory\n", len(checkpoints))
			}
		}
	}

	// Compute metrics for the loaded or initial model
	if useFileCheckpoints {
		currentExactAcc, currentClosenessBins, currentApproxScore = bp.EvaluateMetricsFromCheckpointDir(checkpointDir, getLabels(trainSamples), checkpointBatchSize)
	} else {
		currentExactAcc, currentClosenessBins, currentApproxScore = bp.EvaluateMetricsFromCheckpoints(checkpoints, getLabels(trainSamples))
	}
	currentClosenessQuality = computeClosenessQuality(currentClosenessBins)
	fmt.Printf("Starting with model at generation %d with metrics:\n", startGeneration-1)
	fmt.Printf("  ExactAcc: %.4f\n", currentExactAcc)
	fmt.Printf("  ClosenessBins: %v\n", formatClosenessBins(currentClosenessBins))
	fmt.Printf("  ApproxScore: %.4f\n", currentApproxScore)
	fmt.Printf("  ClosenessQuality: %.4f\n", currentClosenessQuality)

	// Evolution loop
	currentNumModels := initialNumModels
	generationsWithoutImprovement := 0
	numWorkers := int(float64(runtime.NumCPU()) * 0.8)
	runtime.GOMAXPROCS(numWorkers)

	for generation := startGeneration; generation <= maxGenerations; generation++ {
		genStartTime := time.Now()
		fmt.Printf("\n=== Generation %d started at %s with %d models ===\n", generation, genStartTime.Format("2006-01-02 15:04:05"), currentNumModels)

		currentClosenessQuality = computeClosenessQuality(currentClosenessBins)

		var results []ModelResult
		if useMultithreading {
			jobChan := make(chan int, currentNumModels)
			resultChan := make(chan ModelResult, currentNumModels)

			for i := 0; i < numWorkers; i++ {
				go func(workerID int) {
					for job := range jobChan {
						resultChan <- evolveModel(bp, trainSamples, checkpoints, job, generation, workerID)
					}
				}(i)
			}

			for i := 0; i < currentNumModels; i++ {
				jobChan <- i
			}
			close(jobChan)

			for i := 0; i < currentNumModels; i++ {
				results = append(results, <-resultChan)
			}
		} else {
			for i := 0; i < currentNumModels; i++ {
				results = append(results, evolveModel(bp, trainSamples, checkpoints, i, generation, 0))
			}
		}

		numTournaments := 5
		var bestSelected ModelResult
		bestImprovement := -math.MaxFloat64
		for i := 0; i < numTournaments; i++ {
			candidate := tournamentSelection(results, currentExactAcc, currentClosenessQuality, currentApproxScore, 3)
			candidateImprovement := computeTotalImprovement(candidate, currentExactAcc, currentClosenessQuality, currentApproxScore)
			if candidateImprovement > bestImprovement {
				bestImprovement = candidateImprovement
				bestSelected = candidate
			}
		}

		if bestImprovement > 0 {
			newClosenessQuality := computeClosenessQuality(bestSelected.ClosenessBins)
			deltaExactAcc := bestSelected.ExactAcc - currentExactAcc
			deltaApproxScore := bestSelected.ApproxScore - currentApproxScore
			deltaClosenessQuality := newClosenessQuality - currentClosenessQuality

			fmt.Printf("Improved model found in generation %d with total improvement %.4f via tournament selection\n", generation, bestImprovement)
			fmt.Printf("Metric improvements:\n")
			fmt.Printf("  ExactAcc: %.4f → %.4f (Δ %.4f)\n", currentExactAcc, bestSelected.ExactAcc, deltaExactAcc)
			fmt.Printf("  ClosenessBins: %v → %v\n", formatClosenessBins(currentClosenessBins), formatClosenessBins(bestSelected.ClosenessBins))
			fmt.Printf("  ClosenessQuality: %.4f → %.4f (Δ %.4f)\n", currentClosenessQuality, newClosenessQuality, deltaClosenessQuality)
			fmt.Printf("  ApproxScore: %.4f → %.4f (Δ %.4f)\n", currentApproxScore, bestSelected.ApproxScore, deltaApproxScore)

			bp = bestSelected.BP
			currentExactAcc = bestSelected.ExactAcc
			currentClosenessBins = bestSelected.ClosenessBins
			currentApproxScore = bestSelected.ApproxScore

			modelPath := filepath.Join(modelDir, fmt.Sprintf("gen_%d.json", generation))
			if err := saveModel(bp, modelPath); err != nil {
				log.Printf("Failed to save model for generation %d: %v", generation, err)
			}

			if useFileCheckpoints {
				fmt.Println("Recreating checkpoint files with updated model...")
				if err := os.RemoveAll(checkpointDir); err != nil && !os.IsNotExist(err) {
					log.Printf("Failed to clean up checkpoint directory %s: %v", checkpointDir, err)
				}
				if err := bp.SaveCheckpointsToDirectory(getInputs(trainSamples), 1, checkpointDir); err != nil {
					log.Printf("Failed to save new checkpoints: %v", err)
				} else {
					fmt.Printf("New checkpoints saved to %s\n", checkpointDir)
					// Update latest_model.txt
					latestModelFile := filepath.Join(baseCheckpointDir, "latest_model.txt")
					if err := os.WriteFile(latestModelFile, []byte(fmt.Sprintf("%d", generation)), 0644); err != nil {
						log.Printf("Failed to write latest_model.txt: %v", err)
					}
				}
			} else {
				fmt.Println("Recreating checkpoint in memory with updated model...")
				checkpoints = bp.CheckpointAllHiddenNeurons(getInputs(trainSamples), 1)
				fmt.Printf("New checkpoint created with %d samples in memory\n", len(checkpoints))
			}

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

	// Do not clean up checkpointDir to allow resuming
	processEndTime := time.Now()
	totalProcessTime := processEndTime.Sub(processStartTime).Seconds()
	fmt.Printf("\nProcess finished at %s, total time: %.2f seconds\n", processEndTime.Format("2006-01-02 15:04:05"), totalProcessTime)
}

// **selectBestInitialModel** with unique checkpoint directories per model
func selectBestInitialModel(numModels int, layers []int, hiddenAct, outputAct string, samples []Sample, numWorkers int) (*phase.Phase, float64, []float64, float64) {
	type evalResult struct {
		bp            *phase.Phase
		exactAcc      float64
		closenessBins []float64
		approxScore   float64
	}

	jobChan := make(chan *phase.Phase, numModels)
	resultChan := make(chan evalResult, numModels)

	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			for bp := range jobChan {
				var exactAcc float64
				var closenessBins []float64
				var approxScore float64
				if useFileCheckpoints {
					dirPath := filepath.Join(baseCheckpointDir, fmt.Sprintf("initial_model_%d_worker_%d", bp.ID, workerID))
					if err := bp.SaveCheckpointsToDirectory(getInputs(samples), 1, dirPath); err != nil {
						log.Printf("Worker %d failed to save checkpoints for model %d to %s: %v", workerID, bp.ID, dirPath, err)
						continue
					}
					exactAcc, closenessBins, approxScore = bp.EvaluateMetricsFromCheckpointDir(dirPath, getLabels(samples), checkpointBatchSize)
					if err := os.RemoveAll(dirPath); err != nil {
						log.Printf("Worker %d failed to remove checkpoint directory %s: %v", workerID, dirPath, err)
					}
				} else {
					checkpoints := bp.CheckpointAllHiddenNeurons(getInputs(samples), 1)
					exactAcc, closenessBins, approxScore = bp.EvaluateMetricsFromCheckpoints(checkpoints, getLabels(samples))
				}
				resultChan <- evalResult{bp, exactAcc, closenessBins, approxScore}
			}
		}(i)
	}

	for i := 0; i < numModels; i++ {
		bp := phase.NewPhaseWithLayers(layers, hiddenAct, outputAct)
		bp.ID = i + 1
		jobChan <- bp
	}
	close(jobChan)

	bestResult := evalResult{exactAcc: -1}
	for i := 0; i < numModels; i++ {
		result := <-resultChan
		if result.exactAcc > bestResult.exactAcc {
			bestResult = result
		}
	}

	return bestResult.bp, bestResult.exactAcc, bestResult.closenessBins, bestResult.approxScore
}

// **evolveModel** with unique checkpoint directories for each model
func evolveModel(originalBP *phase.Phase, samples []Sample, checkpoints []map[int]map[string]interface{}, copyID int, generation int, workerID int) ModelResult {
	bestBP := originalBP.Copy()
	bestBP.ID = copyID

	var bestExactAcc float64
	var bestClosenessBins []float64
	var bestApproxScore float64
	if useFileCheckpoints {
		dirPath := filepath.Join(baseCheckpointDir, fmt.Sprintf("evolved_gen_%d_id_%d_worker_%d", generation, copyID, workerID))
		checkpointDir := filepath.Join(baseCheckpointDir, sharedCheckpointSubDir)
		if err := bestBP.SaveCheckpointsToDirectory(getInputs(samples), 1, dirPath); err != nil {
			log.Printf("Worker %d failed to save checkpoints for evolved model gen %d id %d to %s: %v", workerID, generation, copyID, dirPath, err)
			bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateMetricsFromCheckpointDir(checkpointDir, getLabels(samples), checkpointBatchSize)
		} else {
			bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateMetricsFromCheckpointDir(dirPath, getLabels(samples), checkpointBatchSize)
			if err := os.RemoveAll(dirPath); err != nil {
				log.Printf("Worker %d failed to remove checkpoint directory %s: %v", workerID, dirPath, err)
			}
		}
	} else {
		bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateMetricsFromCheckpoints(checkpoints, getLabels(samples))
	}
	bestClosenessQuality := computeClosenessQuality(bestClosenessBins)
	neuronsAdded := 0

	iterations := 0
	consecutiveFailures := 0
	maxConsecutiveFailures := 5

	for consecutiveFailures < maxConsecutiveFailures && iterations < maxIterations {
		iterations++

		currentBP := bestBP.Copy()
		numToAdd := rand.Intn(10) + 5
		for i := 0; i < numToAdd; i++ {
			newNeuron := currentBP.AddNeuronFromPreOutputs("dense", "", 1, 50)
			if newNeuron != nil {
				currentBP.AddNewNeuronToOutput(newNeuron.ID)
				neuronsAdded++
			}
		}

		var newExactAcc float64
		var newClosenessBins []float64
		var newApproxScore float64
		if useFileCheckpoints {
			dirPath := filepath.Join(baseCheckpointDir, fmt.Sprintf("evolved_gen_%d_id_%d_iter_%d_worker_%d", generation, copyID, iterations, workerID))
			checkpointDir := filepath.Join(baseCheckpointDir, sharedCheckpointSubDir)
			if err := currentBP.SaveCheckpointsToDirectory(getInputs(samples), 1, dirPath); err != nil {
				log.Printf("Worker %d failed to save checkpoints for evolved model gen %d id %d iter %d to %s: %v", workerID, generation, copyID, iterations, dirPath, err)
				newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateMetricsFromCheckpointDir(checkpointDir, getLabels(samples), checkpointBatchSize)
			} else {
				newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateMetricsFromCheckpointDir(dirPath, getLabels(samples), checkpointBatchSize)
				if err := os.RemoveAll(dirPath); err != nil {
					log.Printf("Worker %d failed to remove checkpoint directory %s: %v", workerID, dirPath, err)
				}
			}
		} else {
			newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateMetricsFromCheckpoints(checkpoints, getLabels(samples))
		}
		newClosenessQuality := computeClosenessQuality(newClosenessBins)

		fmt.Printf("Sandbox %d, Iter %d: eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
			copyID, iterations, newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)

		improvedMetrics := []string{}
		if newExactAcc > bestExactAcc+epsilon {
			improvedMetrics = append(improvedMetrics, "eA")
		}
		if newClosenessQuality > bestClosenessQuality+epsilon {
			improvedMetrics = append(improvedMetrics, "cQ")
		}
		if newApproxScore > bestApproxScore+epsilon {
			improvedMetrics = append(improvedMetrics, "aS")
		}

		if len(improvedMetrics) > 0 {
			fmt.Printf("Sandbox %d: Improvement at Iter %d (%s): eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
				copyID, iterations, strings.Join(improvedMetrics, ", "), newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)
			bestBP = currentBP
			bestExactAcc = newExactAcc
			bestClosenessBins = newClosenessBins
			bestClosenessQuality = newClosenessQuality
			bestApproxScore = newApproxScore
			consecutiveFailures = 0
		} else {
			consecutiveFailures++
		}
	}

	fmt.Printf("Sandbox %d: Exited after %d iterations, %d consecutive failures, eA=%.4f, cQ=%.4f, aS=%.4f\n",
		copyID, iterations, consecutiveFailures, bestExactAcc, bestClosenessQuality, bestApproxScore)
	return ModelResult{
		BP:            bestBP,
		ExactAcc:      bestExactAcc,
		ClosenessBins: bestClosenessBins,
		ApproxScore:   bestApproxScore,
		NeuronsAdded:  neuronsAdded,
	}
}

func getLabels(samples []Sample) []float64 {
	labels := make([]float64, len(samples))
	for i, sample := range samples {
		labels[i] = float64(sample.Label)
	}
	return labels
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
	quality := 0.0
	for i := 5; i < len(bins); i++ {
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
