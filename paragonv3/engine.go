package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"phase" // Replace with your actual import path, e.g., "github.com/you/phase"
)

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"
)

// main demonstrates:
//  1. Downloading MNIST (if needed),
//  2. Loading data as []map[int]float64 + []int labels,
//  3. Building a small network with checkpointing support,
//  4. Demonstrating checkpointing by comparing full forward pass and checkpoint-based outputs with timing,
//  5. Converting labels from int to float64 to call EvaluateMetrics,
//  6. Printing out the metrics (exact accuracy, close accuracies, proximity).
func main() {
	// Seed random number generator (not used for shuffling)
	rand.Seed(time.Now().UnixNano())

	// Step 1: Download MNIST dataset if not already present
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Step 2: Load MNIST training and testing datasets directly into maps and labels
	fmt.Println("Step 2: Loading MNIST training and testing datasets...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	testInputs, testLabels, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading testing MNIST: %v", err)
	}

	// Step 3: Split training data into 80% training and 20% validation without shuffling
	fmt.Println("Step 3: Splitting training data into 80% training and 20% validation without shuffling...")
	totalTrainSamples := 60000
	trainSplit := int(0.8 * float64(totalTrainSamples)) // 48,000 training samples
	valSplit := totalTrainSamples - trainSplit          // 12,000 validation samples

	// No shuffling: take first 48k for training, next 12k for validation
	trainData := trainInputs[:trainSplit]
	trainLbls := trainLabels[:trainSplit]
	valData := trainInputs[trainSplit:totalTrainSamples]
	valLbls := trainLabels[trainSplit:totalTrainSamples]

	_ = trainLbls
	_ = valSplit
	_ = testLabels

	// Print dataset sizes
	fmt.Printf("Training set:   %d samples\n", len(trainData))
	fmt.Printf("Validation set: %d samples\n", len(valData))
	fmt.Printf("Testing set:    %d samples\n", len(testInputs))

	// Step 4: Create a dummy neural network
	fmt.Println("Step 4: Creating a dummy neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	// bp.SetDebug(true) // Uncomment to enable debug prints for detailed output

	// Demonstrate checkpointing feature with timing
	fmt.Println("Demonstrating checkpointing feature...")

	// Select the first 5 validation samples
	numSamples := 5
	if len(valData) < numSamples {
		numSamples = len(valData)
	}
	batchInputs := valData[:numSamples]
	batchLabels := valLbls[:numSamples]

	_ = batchLabels

	// Run full forward pass for each sample and record outputs with timing
	fullOutputsList := make([]map[int]float64, numSamples)
	fullTimes := make([]time.Duration, numSamples) // Store time per sample
	for i, inputMap := range batchInputs {
		startFull := time.Now()
		bp.ResetNeuronValues()  // Reset neuron values before each forward pass
		bp.Forward(inputMap, 1) // 1 timestep for feedforward network
		fullOutputsList[i] = bp.GetOutputs()
		fullTimes[i] = time.Since(startFull)
	}

	// Create checkpoints for the batch with timing
	startCheckpointCreate := time.Now()
	checkpoints := bp.CheckpointPreOutputNeurons(batchInputs, 1)
	checkpointCreateTime := time.Since(startCheckpointCreate)

	// Compute outputs from checkpoints with timing per sample
	checkpointOutputsList := make([]map[int]float64, numSamples)
	checkpointComputeTimes := make([]time.Duration, numSamples) // Store time per sample
	for i, checkpoint := range checkpoints {
		startCheckpointCompute := time.Now()
		bp.ResetNeuronValues() // Reset neuron values before computing from checkpoint
		checkpointOutputsList[i] = bp.ComputeOutputsFromCheckpoint(checkpoint)
		checkpointComputeTimes[i] = time.Since(startCheckpointCompute)
	}

	// Compare the outputs and display times next to each matching neuron
	tolerance := 1e-6
	for i := 0; i < numSamples; i++ {
		fmt.Printf("Sample %d:\n", i)
		fullOutputs := fullOutputsList[i]
		checkpointOutputs := checkpointOutputsList[i]
		// Calculate per-sample checkpoint creation time (approximate)
		checkpointCreateTimePerSample := checkpointCreateTime / time.Duration(numSamples)
		totalCheckpointTime := checkpointCreateTimePerSample + checkpointComputeTimes[i]
		timeSaved := fullTimes[i] - totalCheckpointTime
		for id, fullVal := range fullOutputs {
			checkpointVal, exists := checkpointOutputs[id]
			if !exists || math.Abs(fullVal-checkpointVal) > tolerance {
				fmt.Printf("  Mismatch for output neuron %d: full=%f, checkpoint=%f\n", id, fullVal, checkpointVal)
			} else {
				fmt.Printf("  Output neuron %d matches: %f (Full Time: %d ms) (Checkpoint Runtime: %d ms) (Time Saved: %d ms)\n",
					id, fullVal,
					fullTimes[i].Milliseconds(),
					totalCheckpointTime.Milliseconds(),
					timeSaved.Milliseconds())
			}
		}
	}
	fmt.Println("Checkpointing demonstration complete.")

	// Convert validation labels to float64 for EvaluateMetrics
	valLabelsFloat := make([]float64, len(valLbls))
	for i, lbl := range valLbls {
		valLabelsFloat[i] = float64(lbl)
	}

	// Step 5: Test the EvaluateMetrics function on the validation set
	fmt.Println("Step 5: Testing the EvaluateMetrics function on the validation set...")
	exactAcc, closeAccs, proximity := bp.EvaluateMetrics(valData, valLabelsFloat)

	// Print evaluation results
	fmt.Printf("\nValidation Set Metrics:\n")
	fmt.Printf("Exact Accuracy:  %.2f%%\n", exactAcc)
	fmt.Printf("Proximity Score: %.2f\n", proximity)
	fmt.Println("Close Accuracies at different thresholds:")
	for i, acc := range closeAccs {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: %.2f%%\n", threshold, acc)
	}

	fmt.Println("\nData preparation and evaluation complete.")
}

// ### Helper Functions

// ensureMNISTDownloads downloads and unzips MNIST files if they don’t exist.
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

// loadMNIST loads MNIST images and labels directly into slices of maps and int labels.
// For each image, we create a map[int]float64 of length 784 (28×28 pixels).
// Key = pixel index (0..783), Value = normalized pixel intensity [0..1].
// Labels are integers in [0..9].
func loadMNIST(dir, imageFile, labelFile string, limit int) ([]map[int]float64, []int, error) {
	// Open image file
	imgPath := filepath.Join(dir, imageFile)
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open image file: %w", err)
	}
	defer fImg.Close()

	// Read image header
	var headerImg [16]byte
	if _, err := fImg.Read(headerImg[:]); err != nil {
		return nil, nil, fmt.Errorf("read image header: %w", err)
	}
	numImages := int(binary.BigEndian.Uint32(headerImg[4:8]))
	if limit > numImages {
		limit = numImages
	}

	// Open label file
	lblPath := filepath.Join(dir, labelFile)
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open label file: %w", err)
	}
	defer fLbl.Close()

	// Read label header
	var headerLbl [8]byte
	if _, err := fLbl.Read(headerLbl[:]); err != nil {
		return nil, nil, fmt.Errorf("read label header: %w", err)
	}
	numLabels := int(binary.BigEndian.Uint32(headerLbl[4:8]))
	if limit > numLabels {
		limit = numLabels
	}

	// Initialize slices
	inputs := make([]map[int]float64, limit)
	labels := make([]int, limit)

	// Buffer for one image (784 pixels)
	buf := make([]byte, 784)
	for i := 0; i < limit; i++ {
		// Read image data
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data at sample %d: %w", i, err)
		}
		// Create input map for this sample
		inputMap := make(map[int]float64, 784)
		for px := 0; px < 784; px++ {
			// Normalize [0..255] -> [0..1]
			inputMap[px] = float64(buf[px]) / 255.0
		}
		inputs[i] = inputMap

		// Read label data
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data at sample %d: %w", i, err)
		}
		labels[i] = int(lblByte[0])
	}

	return inputs, labels, nil
}
