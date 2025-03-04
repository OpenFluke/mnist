package main

import (
	"encoding/binary"
	"fmt"
	"log"
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
// 1. Downloading MNIST (if needed),
// 2. Loading data as []map[int]float64 + []int labels,
// 3. Splitting data into training and validation sets,
// 4. Building a small network with checkpointing support,
// 5. Demonstrating checkpointing by comparing full forward pass and computation from checkpoints,
// 6. Evaluating and comparing metrics computation times using EvaluateMetrics and EvaluateMetricsFromCheckpoints.
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
	fmt.Println("Step 3: Splitting training data into 80% training and 20% validation without shuffling...")
	totalTrainSamples := 60000
	trainSplit := int(0.8 * float64(totalTrainSamples))
	valSplit := totalTrainSamples - trainSplit

	trainData := trainInputs[:trainSplit]
	trainLbls := trainLabels[:trainSplit]
	valData := trainInputs[trainSplit:totalTrainSamples]
	valLbls := trainLabels[trainSplit:totalTrainSamples]

	_ = trainLbls
	_ = valSplit
	_ = testLabels

	fmt.Printf("Training set:   %d samples\n", len(trainData))
	fmt.Printf("Validation set: %d samples\n", len(valData))
	fmt.Printf("Testing set:    %d samples\n", len(testInputs))

	// Step 4: Create a dummy neural network
	fmt.Println("Step 4: Creating a dummy neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")

	// Demonstrate checkpointing feature
	fmt.Println("Demonstrating checkpointing feature on the entire training set...")
	batchInputs := trainData
	numSamples := len(batchInputs)

	fmt.Println("Creating checkpoints (preprocessing step)...")
	startCreate := time.Now()
	checkpoints := bp.CheckpointPreOutputNeurons(batchInputs, 1)
	createTime := time.Since(startCreate)
	fmt.Printf("Checkpoint Creation Time: %d ms\n", createTime.Milliseconds())

	startFull := time.Now()
	for _, inputMap := range batchInputs {
		bp.ResetNeuronValues()
		bp.Forward(inputMap, 1)
		_ = bp.GetOutputs()
	}
	fullTimeTotal := time.Since(startFull)
	fullTimePerSample := fullTimeTotal / time.Duration(numSamples)

	startCompute := time.Now()
	for _, checkpoint := range checkpoints {
		bp.ResetNeuronValues()
		_ = bp.ComputeOutputsFromCheckpoint(checkpoint)
	}
	computeTimeTotal := time.Since(startCompute)
	computeTimePerSample := computeTimeTotal / time.Duration(numSamples)

	timeSavedTotal := fullTimeTotal - computeTimeTotal
	timeSavedPerSample := fullTimePerSample - computeTimePerSample

	fmt.Println("\nTiming Summary for Checkpointing Demonstration:")
	fmt.Printf("Total Full Forward Pass Time: %d ms\n", fullTimeTotal.Milliseconds())
	fmt.Printf("Average Full Forward Pass Time per Sample: %d μs\n", fullTimePerSample.Microseconds())
	fmt.Printf("Total Computation Time from Checkpoints: %d ms\n", computeTimeTotal.Milliseconds())
	fmt.Printf("Average Computation Time from Checkpoints per Sample: %d μs\n", computeTimePerSample.Microseconds())
	fmt.Printf("Total Time Saved: %d ms\n", timeSavedTotal.Milliseconds())
	fmt.Printf("Average Time Saved per Sample: %d μs\n", timeSavedPerSample.Microseconds())

	fmt.Println("Checkpointing demonstration complete.")

	// Convert validation labels to float64 for EvaluateMetrics
	valLabelsFloat := make([]float64, len(valLbls))
	for i, lbl := range valLbls {
		valLabelsFloat[i] = float64(lbl)
	}

	// Step 5: Test and compare EvaluateMetrics vs EvaluateMetricsFromCheckpoints
	fmt.Println("\nStep 5: Comparing EvaluateMetrics and EvaluateMetricsFromCheckpoints on the validation set...")

	// Part 1: Evaluate using full forward pass
	fmt.Println("Evaluating with full forward pass (EvaluateMetrics)...")
	startFullEval := time.Now()
	exactAccFull, closeAccsFull, proximityFull := bp.EvaluateMetrics(valData, valLabelsFloat)
	fullEvalTime := time.Since(startFullEval)

	fmt.Printf("\nValidation Set Metrics (Full Forward Pass):\n")
	fmt.Printf("Exact Accuracy:  %.2f%%\n", exactAccFull)
	fmt.Printf("Proximity Score: %.2f\n", proximityFull)
	fmt.Println("Close Accuracies at different thresholds:")
	for i, acc := range closeAccsFull {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: %.2f%%\n", threshold, acc)
	}
	fmt.Printf("Time taken for full evaluation: %d ms\n", fullEvalTime.Milliseconds())

	// Part 2: Compute checkpoints for validation set
	fmt.Println("\nComputing checkpoints for the validation set...")
	startCheckpoint := time.Now()
	valCheckpoints := bp.CheckpointPreOutputNeurons(valData, 1)
	checkpointTime := time.Since(startCheckpoint)
	fmt.Printf("Time taken to compute checkpoints: %d ms\n", checkpointTime.Milliseconds())

	// Part 3: Evaluate using checkpoints
	fmt.Println("Evaluating with checkpoints (EvaluateMetricsFromCheckpoints)...")
	startCheckpointEval := time.Now()
	exactAccCheckpoint, closeAccsCheckpoint, proximityCheckpoint := bp.EvaluateMetricsFromCheckpoints(valCheckpoints, valLabelsFloat)
	checkpointEvalTime := time.Since(startCheckpointEval)

	fmt.Printf("\nValidation Set Metrics (From Checkpoints):\n")
	fmt.Printf("Exact Accuracy:  %.2f%%\n", exactAccCheckpoint)
	fmt.Printf("Proximity Score: %.2f\n", proximityCheckpoint)
	fmt.Println("Close Accuracies at different thresholds:")
	for i, acc := range closeAccsCheckpoint {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: %.2f%%\n", threshold, acc)
	}
	fmt.Printf("Time taken for evaluation from checkpoints: %d ms\n", checkpointEvalTime.Milliseconds())

	// Part 4: Compare results
	fmt.Println("\nComparison of Metrics:")
	fmt.Printf("Exact Accuracy (Full): %.2f%% vs (Checkpoint): %.2f%%\n", exactAccFull, exactAccCheckpoint)
	fmt.Printf("Proximity Score (Full): %.2f vs (Checkpoint): %.2f\n", proximityFull, proximityCheckpoint)
	fmt.Println("Closeness Bins Comparison:")
	for i := range closeAccsFull {
		threshold := (i + 1) * 10
		fmt.Printf("  %d%% Threshold: (Full) %.2f%% vs (Checkpoint) %.2f%%\n", threshold, closeAccsFull[i], closeAccsCheckpoint[i])
	}

	fmt.Println("\nTiming Comparison:")
	fmt.Printf("Full Evaluation Time: %d ms\n", fullEvalTime.Milliseconds())
	fmt.Printf("Checkpoint Evaluation Time: %d ms\n", checkpointEvalTime.Milliseconds())
	fmt.Printf("Time Saved (Evaluation Only): %d ms\n", (fullEvalTime - checkpointEvalTime).Milliseconds())
	fmt.Printf("Checkpoint Creation + Evaluation Time: %d ms\n", (checkpointTime + checkpointEvalTime).Milliseconds())

	fmt.Println("\nData preparation, evaluation, and comparison complete.")
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
