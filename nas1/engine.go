package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"paragon"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
	// Checkpoint directories: one file per input.
	checkpointTrainDir  = "checkpoints/train"
	checkpointSampleDir = "checkpoints/test"
	// We take the checkpoint from layer index 1.
	checkpointLayerIdx = 1
)

func main() {
	// ----------------- Data Preparation -----------------
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST downloads: %v", err)
	}
	fmt.Println("MNIST data ready.")

	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	// For this example we use the 80% training split for training/checkpointing.
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Printf("Training samples: %d, Test samples: %d\n", len(trainSetInputs), len(testInputs))

	// ----------------- Model Creation / Loading -----------------
	// Define network architecture.
	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	modelPath := filepath.Join(modelDir, modelFile)
	var nn *paragon.Network
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("Pre-trained model found. Loading model...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
		if err := nn.LoadFromJSON(modelPath); err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
	} else {
		fmt.Println("No pre-trained model found. Training new model...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
		trainer := paragon.Trainer{
			Network: nn,
			Config: paragon.TrainConfig{
				Epochs:           5,
				LearningRate:     0.01,
				PlateauThreshold: 0.001,
				PlateauLimit:     3,
				EarlyStopAcc:     0.95,
				Debug:            true,
			},
		}
		trainer.TrainWithValidation(trainSetInputs, trainSetTargets, nil, nil, testInputs, testTargets)
		if err := nn.SaveToJSON(modelPath); err != nil {
			log.Fatalf("Failed to save model: %v", err)
		}
		fmt.Println("Model trained and saved.")
	}

	// ----------------- Create Checkpoint Files -----------------
	// Create directories.
	if err := os.MkdirAll(checkpointTrainDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create training checkpoint directory: %v", err)
	}
	if err := os.MkdirAll(checkpointSampleDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create sample checkpoint directory: %v", err)
	}

	// For each training input.
	fmt.Println("Saving individual checkpoint files for training data...")
	for i, input := range trainSetInputs {
		filename := filepath.Join(checkpointTrainDir, fmt.Sprintf("train_cp_%d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			nn.Forward(input)
			cpState := nn.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal training checkpoint for sample %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write training checkpoint file for sample %d: %v", i, err)
			}
		}
	}

	// For each test input (sample data).
	fmt.Println("Saving individual checkpoint files for test data...")
	for i, input := range testInputs {
		filename := filepath.Join(checkpointSampleDir, fmt.Sprintf("test_cp_%d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			nn.Forward(input)
			cpState := nn.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal test checkpoint for sample %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write test checkpoint file for sample %d: %v", i, err)
			}
		}
	}

	// ----------------- Evaluation of Original Model -----------------
	// Full forward evaluation on training set.
	var expectedTrain, predictionsTrain []float64
	for i, input := range trainSetInputs {
		nn.Forward(input)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(trainSetTargets[i][0])
		expectedTrain = append(expectedTrain, float64(trueLabel))
		predictionsTrain = append(predictionsTrain, float64(pred))
	}
	nn.EvaluateModel(expectedTrain, predictionsTrain)
	trainAccuracy := computeAccuracy(predictionsTrain, expectedTrain)
	fmt.Printf("Training Full Forward: Accuracy: %.2f%%, ADHD Score: %.4f\n", trainAccuracy*100, nn.Performance.Score)

	// Checkpoint evaluation on training set.
	var expectedTrainCP, predictionsTrainCP []float64
	trainCPFiles := sortedFilesInDir(checkpointTrainDir)
	for i, cpFile := range trainCPFiles {
		data, err := os.ReadFile(cpFile)
		if err != nil {
			log.Fatalf("Failed to read training checkpoint file %s: %v", cpFile, err)
		}
		var cpState [][]float64
		if err := json.Unmarshal(data, &cpState); err != nil {
			log.Fatalf("Failed to unmarshal training checkpoint file %s: %v", cpFile, err)
		}
		nn.ForwardFromLayer(checkpointLayerIdx, cpState)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(trainSetTargets[i][0])
		expectedTrainCP = append(expectedTrainCP, float64(trueLabel))
		predictionsTrainCP = append(predictionsTrainCP, float64(pred))
	}
	nn.EvaluateModel(expectedTrainCP, predictionsTrainCP)
	fmt.Printf("Training Checkpoint Evaluation: ADHD Score: %.4f\n", nn.Performance.Score)

	// ----------------- Explore Appending a New Layer -----------------
	// We now try adding an extra layer after the checkpoint layer to see if the ADHD score improves.
	// We pass the training checkpoint files and targets along with the base network parameters.
	exploreAppendedLayersBatchesWithCheckpoint(nn, checkpointTrainDir, trainSetTargets, layerSizes, activations, fullyConnected, 16)

	fmt.Println("Evaluation complete.")
}

// exploreAppendedLayersBatches limits concurrency to 80% of CPU cores while processing candidate models in batches.
// For each candidate (an appended layer configuration), it uses EvaluateFromCheckpoint to compute the ADHD score
// starting from the checkpoint.
func exploreAppendedLayersBatchesWithCheckpoint(baseModel *paragon.Network, cpDir string, targets [][][]float64,
	layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool, batchSize int) {

	// Read and sort checkpoint file paths.
	cpFiles := sortedFilesInDir(cpDir)
	if len(cpFiles) == 0 {
		log.Fatalf("No checkpoint files found in %s", cpDir)
	}

	// Define candidate configuration struct.
	type candidate struct {
		width      int
		height     int
		activation string
	}

	// Generate many candidate configurations.
	var candidates []candidate
	widths := []int{8, 16, 32, 64}
	heights := []int{8, 16, 32, 64}
	acts := []string{"leaky_relu", "relu", "sigmoid", "tanh"}
	for _, w := range widths {
		for _, h := range heights {
			for _, a := range acts {
				candidates = append(candidates, candidate{width: w, height: h, activation: a})
			}
		}
	}
	fmt.Printf("\nExploring %d appended layer configurations in batches of %d on training checkpoints:\n", len(candidates), batchSize)

	// Structure to hold candidate results.
	type candidateResult struct {
		width      int
		height     int
		activation string
		adhdScore  float64
	}

	// Create a channel to collect candidate results.
	results := make(chan candidateResult, len(candidates))

	// Limit concurrency to 80% of available CPU cores.
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * 0.8)
	if numThreads < 1 {
		numThreads = 1
	}
	sem := make(chan struct{}, numThreads)

	// Process candidates in batches.
	for i := 0; i < len(candidates); i += batchSize {
		endIndex := i + batchSize
		if endIndex > len(candidates) {
			endIndex = len(candidates)
		}
		currentBatch := candidates[i:endIndex]
		var wg sync.WaitGroup

		for _, cand := range currentBatch {
			wg.Add(1)
			// Acquire a semaphore token.
			sem <- struct{}{}
			go func(c candidate) {
				defer wg.Done()
				defer func() { <-sem }()
				// Clone the base model.
				cloned := cloneNetwork(baseModel, layerSizes, activations, fullyConnected)
				// Append a new layer at index 2 (i.e. between checkpoint layer and output).
				cloned.AddLayer(2, c.width, c.height, c.activation, true)
				fmt.Printf("Evaluating Candidate: width=%d, height=%d, activation=%s\n", c.width, c.height, c.activation)

				// Collect all checkpoint states and expected outputs.
				var allCPStates [][][]float64
				var expected []float64
				for j, cpFile := range cpFiles {
					data, err := os.ReadFile(cpFile)
					if err != nil {
						log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
					}
					var cpState [][]float64
					if err := json.Unmarshal(data, &cpState); err != nil {
						log.Fatalf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
					}
					allCPStates = append(allCPStates, cpState)
					trueLabel := paragon.ArgMax(targets[j][0])
					expected = append(expected, float64(trueLabel))
				}

				// Use EvaluateFromCheckpoint to compute the ADHD score starting from the checkpoint.
				cloned.EvaluateFromCheckpoint(allCPStates, expected, checkpointLayerIdx)
				candidateScore := cloned.Performance.Score
				fmt.Printf("Candidate (w=%d, h=%d, act=%s) evaluated: ADHD Score = %.4f\n", c.width, c.height, c.activation, candidateScore)
				results <- candidateResult{
					width:      c.width,
					height:     c.height,
					activation: c.activation,
					adhdScore:  candidateScore,
				}
			}(cand)
		}
		wg.Wait()
		fmt.Printf("Completed batch for candidates %d to %d\n", i, endIndex-1)
	}
	close(results)

	// Print out the final candidate results.
	fmt.Println("\nFinal Candidate Results:")
	for res := range results {
		fmt.Printf("Candidate (w=%d, h=%d, act=%s): ADHD Score = %.4f\n", res.width, res.height, res.activation, res.adhdScore)
	}
}

// exploreAppendedLayersBatches limits concurrency to 80% of CPU cores while processing candidate models in batches.
// For each candidate (an appended layer configuration), it evaluates the ADHD score using checkpoint files
// and prints the score immediately as well as in a final summary.
func exploreAppendedLayersBatches(baseModel *paragon.Network, cpDir string, targets [][][]float64,
	layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool, batchSize int) {

	// Read and sort checkpoint file paths.
	cpFiles := sortedFilesInDir(cpDir)
	if len(cpFiles) == 0 {
		log.Fatalf("No checkpoint files found in %s", cpDir)
	}

	// Define candidate configuration struct.
	type candidate struct {
		width      int
		height     int
		activation string
	}

	// Generate many candidate configurations.
	var candidates []candidate
	widths := []int{8, 16, 32, 64}
	heights := []int{8, 16, 32, 64}
	acts := []string{"leaky_relu", "relu", "sigmoid", "tanh"}
	for _, w := range widths {
		for _, h := range heights {
			for _, a := range acts {
				candidates = append(candidates, candidate{width: w, height: h, activation: a})
			}
		}
	}
	fmt.Printf("\nExploring %d appended layer configurations in batches of %d on training checkpoints:\n", len(candidates), batchSize)

	// Structure to hold candidate results.
	type candidateResult struct {
		width      int
		height     int
		activation string
		adhdScore  float64
	}

	// Create a channel to collect candidate results.
	results := make(chan candidateResult, len(candidates))

	// Calculate allowed concurrent goroutines (80% of available CPU cores).
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * 0.8)
	if numThreads < 1 {
		numThreads = 1
	}
	sem := make(chan struct{}, numThreads)

	// Process candidates in batches.
	for i := 0; i < len(candidates); i += batchSize {
		endIndex := i + batchSize
		if endIndex > len(candidates) {
			endIndex = len(candidates)
		}
		currentBatch := candidates[i:endIndex]
		var wg sync.WaitGroup

		for _, cand := range currentBatch {
			wg.Add(1)
			// Acquire a semaphore token.
			sem <- struct{}{}
			go func(c candidate) {
				defer wg.Done()
				defer func() { <-sem }()
				// Clone the base model.
				cloned := cloneNetwork(baseModel, layerSizes, activations, fullyConnected)
				// Append a new layer at index 2 (i.e. between checkpoint layer and output).
				cloned.AddLayer(2, c.width, c.height, c.activation, true)
				fmt.Printf("Evaluating Candidate: width=%d, height=%d, activation=%s\n", c.width, c.height, c.activation)

				var expected []float64
				var actual []float64
				// Process each checkpoint file sequentially.
				for j, cpFile := range cpFiles {
					data, err := os.ReadFile(cpFile)
					if err != nil {
						log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
					}
					var cpState [][]float64
					if err := json.Unmarshal(data, &cpState); err != nil {
						log.Fatalf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
					}
					// Complete the forward pass from the checkpoint.
					cloned.ForwardFromLayer(checkpointLayerIdx, cpState)
					out := extractOutput(cloned)
					pred := paragon.ArgMax(out)
					trueLabel := paragon.ArgMax(targets[j][0])
					expected = append(expected, float64(trueLabel))
					actual = append(actual, float64(pred))
				}
				cloned.EvaluateModel(expected, actual)
				candidateScore := cloned.Performance.Score
				// Print the candidate result immediately.
				fmt.Printf("Candidate (w=%d, h=%d, act=%s) evaluated: ADHD Score = %.4f\n", c.width, c.height, c.activation, candidateScore)
				results <- candidateResult{
					width:      c.width,
					height:     c.height,
					activation: c.activation,
					adhdScore:  candidateScore,
				}
			}(cand)
		}
		wg.Wait()
		fmt.Printf("Completed batch for candidates %d to %d\n", i, endIndex-1)
	}
	close(results)

	// Print out the final candidate results.
	fmt.Println("\nFinal Candidate Results:")
	for res := range results {
		fmt.Printf("Candidate (w=%d, h=%d, act=%s): ADHD Score = %.4f\n", res.width, res.height, res.activation, res.adhdScore)
	}
}

func exploreAppendedLayersMultiThreadedGenerateCandidate(baseModel *paragon.Network, cpDir string, targets [][][]float64,
	layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool) {

	// Read and sort checkpoint file paths.
	cpFiles := sortedFilesInDir(cpDir)
	if len(cpFiles) == 0 {
		log.Fatalf("No checkpoint files found in %s", cpDir)
	}

	// Define candidate configuration struct.
	type candidate struct {
		width      int
		height     int
		activation string
	}

	// Generate many candidate configurations.
	var candidates []candidate
	widths := []int{8, 16, 32, 64}
	heights := []int{8, 16, 32, 64}
	acts := []string{"leaky_relu", "relu", "sigmoid", "tanh"}
	for _, w := range widths {
		for _, h := range heights {
			for _, a := range acts {
				candidates = append(candidates, candidate{width: w, height: h, activation: a})
			}
		}
	}
	fmt.Printf("\nExploring %d appended layer configurations on training checkpoints:\n", len(candidates))

	// Structure to hold candidate results.
	type candidateResult struct {
		width      int
		height     int
		activation string
		adhdScore  float64
	}

	// Create a channel to collect candidate results.
	results := make(chan candidateResult, len(candidates))
	var wg sync.WaitGroup

	// Process each candidate concurrently.
	for _, cand := range candidates {
		wg.Add(1)
		go func(c candidate) {
			defer wg.Done()
			// Clone the base model using our cloneNetwork function.
			cloned := cloneNetwork(baseModel, layerSizes, activations, fullyConnected)
			// Append a new layer at index 2 (between checkpoint layer and output).
			cloned.AddLayer(2, c.width, c.height, c.activation, true)
			fmt.Printf("Evaluating Candidate: width=%d, height=%d, activation=%s\n", c.width, c.height, c.activation)

			var expected []float64
			var actual []float64
			for i, cpFile := range cpFiles {
				data, err := os.ReadFile(cpFile)
				if err != nil {
					log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
				}
				var cpState [][]float64
				if err := json.Unmarshal(data, &cpState); err != nil {
					log.Fatalf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
				}
				// Use the cloned candidate network to complete the forward pass.
				cloned.ForwardFromLayer(checkpointLayerIdx, cpState)
				out := extractOutput(cloned)
				pred := paragon.ArgMax(out)
				trueLabel := paragon.ArgMax(targets[i][0])
				expected = append(expected, float64(trueLabel))
				actual = append(actual, float64(pred))
			}
			cloned.EvaluateModel(expected, actual)
			results <- candidateResult{
				width:      c.width,
				height:     c.height,
				activation: c.activation,
				adhdScore:  cloned.Performance.Score,
			}
		}(cand)
	}

	wg.Wait()
	close(results)

	// Print out all candidate results.
	for res := range results {
		fmt.Printf("Candidate (w=%d, h=%d, act=%s): ADHD Score = %.4f\n", res.width, res.height, res.activation, res.adhdScore)
	}
}

// exploreAppendedLayers creates candidate models by cloning the base model, appending a new layer with candidate parameters,
// and then evaluating the ADHD score via checkpoint files. The candidate evaluations are run concurrently.
func exploreAppendedLayersMultiThreaded(baseModel *paragon.Network, cpDir string, targets [][][]float64,
	layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool) {

	// Read and sort checkpoint file paths.
	cpFiles := sortedFilesInDir(cpDir)
	if len(cpFiles) == 0 {
		log.Fatalf("No checkpoint files found in %s", cpDir)
	}

	// Define candidate configurations for the appended layer.
	type candidate struct {
		width      int
		height     int
		activation string
	}
	candidates := []candidate{
		{8, 8, "leaky_relu"},
		{16, 16, "leaky_relu"},
		{32, 32, "relu"},
	}

	fmt.Println("\nExploring appended layer configurations on training checkpoints:")

	// Structure to hold the result of each candidate evaluation.
	type candidateResult struct {
		width      int
		height     int
		activation string
		adhdScore  float64
	}

	// Create a channel to collect candidate results.
	results := make(chan candidateResult, len(candidates))
	var wg sync.WaitGroup

	// Process each candidate concurrently.
	for _, cand := range candidates {
		wg.Add(1)
		go func(c candidate) {
			defer wg.Done()
			// Clone the base model using our cloneNetwork function.
			cloned := cloneNetwork(baseModel, layerSizes, activations, fullyConnected)
			// Append a new layer at index 2 (i.e. between checkpoint layer and output).
			cloned.AddLayer(2, c.width, c.height, c.activation, true)
			fmt.Printf("Evaluating Candidate: width=%d, height=%d, activation=%s\n", c.width, c.height, c.activation)

			// Evaluate the candidate using checkpoint files.
			var expected []float64
			var actual []float64
			for i, cpFile := range cpFiles {
				data, err := os.ReadFile(cpFile)
				if err != nil {
					log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
				}
				var cpState [][]float64
				if err := json.Unmarshal(data, &cpState); err != nil {
					log.Fatalf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
				}
				// Use the cloned candidate network to complete the forward pass.
				cloned.ForwardFromLayer(checkpointLayerIdx, cpState)
				out := extractOutput(cloned)
				pred := paragon.ArgMax(out)
				trueLabel := paragon.ArgMax(targets[i][0])
				expected = append(expected, float64(trueLabel))
				actual = append(actual, float64(pred))
			}
			cloned.EvaluateModel(expected, actual)
			result := candidateResult{
				width:      c.width,
				height:     c.height,
				activation: c.activation,
				adhdScore:  cloned.Performance.Score,
			}
			results <- result
		}(cand)
	}

	// Wait for all candidate evaluations to finish.
	wg.Wait()
	close(results)

	// Print out the results.
	for res := range results {
		fmt.Printf("Candidate (w=%d, h=%d, act=%s): ADHD Score = %.4f\n", res.width, res.height, res.activation, res.adhdScore)
	}
}

// exploreAppendedLayers creates candidate models by cloning the base model, appending a new layer with candidate parameters,
// and then evaluating the ADHD score via checkpoint files.
func exploreAppendedLayers(baseModel *paragon.Network, cpDir string, targets [][][]float64,
	layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool) {

	// Read and sort checkpoint file paths.
	cpFiles := sortedFilesInDir(cpDir)
	if len(cpFiles) == 0 {
		log.Fatalf("No checkpoint files found in %s", cpDir)
	}

	// Define candidate configurations for the appended layer.
	candidates := []struct {
		width      int
		height     int
		activation string
	}{
		{8, 8, "leaky_relu"},
		{16, 16, "leaky_relu"},
		{32, 32, "relu"},
	}

	fmt.Println("\nExploring appended layer configurations on training checkpoints:")
	for _, cand := range candidates {
		// Clone the base model (using our updated cloneNetwork which removes problematic fields).
		cloned := cloneNetwork(baseModel, layerSizes, activations, fullyConnected)
		// Append a new layer at index 2 (i.e. between checkpoint layer and output).
		cloned.AddLayer(2, cand.width, cand.height, cand.activation, true)
		fmt.Printf("Candidate: width=%d, height=%d, activation=%s\n", cand.width, cand.height, cand.activation)

		// Evaluate using checkpoint files.
		var expected []float64
		var actual []float64
		for i, cpFile := range cpFiles {
			data, err := os.ReadFile(cpFile)
			if err != nil {
				log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
			}
			var cpState [][]float64
			if err := json.Unmarshal(data, &cpState); err != nil {
				log.Fatalf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
			}
			cloned.ForwardFromLayer(checkpointLayerIdx, cpState)
			out := extractOutput(cloned)
			pred := paragon.ArgMax(out)
			trueLabel := paragon.ArgMax(targets[i][0])
			expected = append(expected, float64(trueLabel))
			actual = append(actual, float64(pred))
		}
		cloned.EvaluateModel(expected, actual)
		fmt.Printf("Candidate (w=%d, h=%d, act=%s): ADHD Score = %.4f\n", cand.width, cand.height, cand.activation, cloned.Performance.Score)
	}
}

// cloneNetwork creates a deep copy of a network via JSON marshalling, after zeroing the Performance field
// (which might contain unsupported +Inf values). After unmarshalling, Performance is reinitialized.
func cloneNetwork(n *paragon.Network, layerSizes []struct{ Width, Height int },
	activations []string, fullyConnected []bool) *paragon.Network {
	// Make a shallow copy and set Performance to nil to avoid marshalling +Inf values.
	temp := *n
	temp.Performance = nil

	data, err := json.Marshal(temp)
	if err != nil {
		log.Fatalf("cloneNetwork: failed to marshal: %v", err)
	}
	clone := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	if err := json.Unmarshal(data, clone); err != nil {
		log.Fatalf("cloneNetwork: failed to unmarshal: %v", err)
	}
	clone.Performance = paragon.NewADHDPerformance()
	return clone
}

// sortedFilesInDir returns a sorted list of file paths from the specified directory.
func sortedFilesInDir(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		log.Fatalf("Failed to read directory %s: %v", dir, err)
	}
	var files []string
	for _, entry := range entries {
		if !entry.IsDir() {
			files = append(files, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(files)
	return files
}

// ----------------- Helper Functions (Data I/O, Evaluation) -----------------

func ensureMNISTDownloads(targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", targetDir, err)
	}
	files := []struct {
		compressed   string
		uncompressed string
	}{
		{"train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"},
		{"train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"},
		{"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
		{"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"},
	}
	for _, f := range files {
		compressedPath := filepath.Join(targetDir, f.compressed)
		uncompressedPath := filepath.Join(targetDir, f.uncompressed)
		if _, err := os.Stat(uncompressedPath); os.IsNotExist(err) {
			if _, err := os.Stat(compressedPath); os.IsNotExist(err) {
				fmt.Printf("Downloading %s...\n", f.compressed)
				if err := downloadFile(baseURL+f.compressed, compressedPath); err != nil {
					return fmt.Errorf("failed to download %s: %w", f.compressed, err)
				}
			}
			fmt.Printf("Unzipping %s...\n", f.compressed)
			if err := unzipFile(compressedPath, uncompressedPath); err != nil {
				return fmt.Errorf("failed to unzip %s: %w", f.compressed, err)
			}
			if err := os.Remove(compressedPath); err != nil {
				log.Printf("Warning: failed to remove %s: %v", compressedPath, err)
			}
		}
	}
	return nil
}

func downloadFile(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func unzipFile(src, dest string) error {
	fSrc, err := os.Open(src)
	if err != nil {
		return err
	}
	defer fSrc.Close()
	gzReader, err := gzip.NewReader(fSrc)
	if err != nil {
		return err
	}
	defer gzReader.Close()
	fDest, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer fDest.Close()
	_, err = io.Copy(fDest, gzReader)
	return err
}

func loadMNISTData(dir string, isTraining bool) ([][][]float64, [][][]float64, error) {
	prefix := "train"
	if !isTraining {
		prefix = "t10k"
	}
	imgPath := filepath.Join(dir, prefix+"-images-idx3-ubyte")
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer fImg.Close()
	var imgHeader [16]byte
	if _, err := fImg.Read(imgHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("failed to read image header: %w", err)
	}
	if magic := binary.BigEndian.Uint32(imgHeader[0:4]); magic != 2051 {
		return nil, nil, fmt.Errorf("invalid image magic number: %d", magic)
	}
	numImages := int(binary.BigEndian.Uint32(imgHeader[4:8]))
	rows := int(binary.BigEndian.Uint32(imgHeader[8:12]))
	cols := int(binary.BigEndian.Uint32(imgHeader[12:16]))
	inputs := make([][][]float64, numImages)

	lblPath := filepath.Join(dir, prefix+"-labels-idx1-ubyte")
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open label file: %w", err)
	}
	defer fLbl.Close()
	var lblHeader [8]byte
	if _, err := fLbl.Read(lblHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("failed to read label header: %w", err)
	}
	_ = int(binary.BigEndian.Uint32(lblHeader[4:8])) // Number of labels (unused here)
	targets := make([][][]float64, numImages)
	imgBuf := make([]byte, rows*cols)
	for i := 0; i < numImages; i++ {
		if _, err := fImg.Read(imgBuf); err != nil {
			return nil, nil, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			img[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				img[r][c] = float64(imgBuf[r*cols+c]) / 255.0
			}
		}
		inputs[i] = img
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("failed to read label %d: %w", i, err)
		}
		targets[i] = labelToTarget(int(lblByte[0]))
	}
	return inputs, targets, nil
}

func labelToTarget(label int) [][]float64 {
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	if label >= 0 && label < 10 {
		target[0][label] = 1.0
	}
	return target
}

func extractOutput(nn *paragon.Network) []float64 {
	outWidth := nn.Layers[nn.OutputLayer].Width
	output := make([]float64, outWidth)
	for x := 0; x < outWidth; x++ {
		output[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
	}
	return output
}

func computeAccuracy(predicted, trueLabels []float64) float64 {
	if len(predicted) != len(trueLabels) {
		return 0
	}
	correct := 0
	for i := range predicted {
		if int(predicted[i]) == int(trueLabels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(predicted))
}
