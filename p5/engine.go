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
//  4. Demonstrating checkpointing by comparing full forward pass and computation from precomputed checkpoints,
//  5. Converting labels from int to float64 to call EvaluateMetrics,
//  6. Printing out the metrics (exact accuracy, close accuracies, proximity).
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

	// New Step: Demonstrate adding a random neuron and comparing checkpoint vs. full forward pass
	fmt.Println("\nDemonstrating adding a random neuron and comparing checkpoint vs. full forward pass...")

	// Make a copy of the original network
	modifiedBP := bp.Copy()

	// Add a new neuron connected from pre-output neurons
	fmt.Println("Adding a new neuron to the modified network...")
	newNeuron := modifiedBP.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
	if newNeuron == nil {
		log.Fatalf("Failed to add new neuron")
	}
	fmt.Printf("Added neuron with ID %d\n", newNeuron.ID)

	// Test on a subset of inputs for demonstration (e.g., first 5)
	numTestSamples := 5
	if numTestSamples > len(batchInputs) {
		numTestSamples = len(batchInputs)
	}

	fmt.Printf("Comparing outputs for %d test samples...\n", numTestSamples)
	for i := 0; i < numTestSamples; i++ {
		inputMap := batchInputs[i]
		checkpoint := checkpoints[i]

		// Method 1: Compute outputs using checkpoint
		modifiedBP.ResetNeuronValues()

		// Set original pre-output neuron states from checkpoint
		for id, state := range checkpoint {
			if neuron, exists := modifiedBP.Neurons[id]; exists {
				modifiedBP.SetNeuronState(neuron, state)
			}
		}

		// Compute the new neuron's value based on checkpointed states
		newNeuronValue := newNeuron.Bias
		for _, conn := range newNeuron.Connections {
			sourceID := int(conn[0])
			weight := conn[1]
			if sourceNeuron, exists := modifiedBP.Neurons[sourceID]; exists {
				newNeuronValue += sourceNeuron.Value * weight
			}
		}
		newNeuron.Value = modifiedBP.ApplyScalarActivation(newNeuronValue, newNeuron.Activation)

		// Compute output neurons manually
		checkpointOutputs := make([]float64, len(modifiedBP.OutputNodes))
		for idx, id := range modifiedBP.OutputNodes {
			neuron := modifiedBP.Neurons[id]
			sum := neuron.Bias
			for _, conn := range neuron.Connections {
				sourceID := int(conn[0])
				weight := conn[1]
				if sourceNeuron, exists := modifiedBP.Neurons[sourceID]; exists {
					sum += sourceNeuron.Value * weight
				}
			}
			checkpointOutputs[idx] = modifiedBP.ApplyScalarActivation(sum, neuron.Activation)
		}

		// Method 2: Compute outputs using full forward pass
		modifiedBP.ResetNeuronValues()
		modifiedBP.Forward(inputMap, 1)
		fullOutputs := modifiedBP.GetOutputs()

		// Compare outputs
		fmt.Printf("\nSample %d:\n", i)
		fmt.Printf("Checkpoint Outputs: %v\n", checkpointOutputs)
		fmt.Printf("Full Pass Outputs:  %v\n", fullOutputs)

		// Check if outputs are approximately equal (within a small tolerance)
		outputsMatch := true
		const tolerance = 1e-5
		for j := range checkpointOutputs {
			if math.Abs(checkpointOutputs[j]-fullOutputs[j]) > tolerance {
				outputsMatch = false
				break
			}
		}
		if outputsMatch {
			fmt.Println("Outputs match within tolerance.")
		} else {
			fmt.Println("Outputs differ beyond tolerance.")
		}
	}

	fmt.Println("Neuron addition and comparison demonstration complete.")

	// Convert validation labels to float64 for EvaluateMetrics
	valLabelsFloat := make([]float64, len(valLbls))
	for i, lbl := range valLbls {
		valLabelsFloat[i] = float64(lbl)
	}

	// Step 5: Test the EvaluateMetrics function
	fmt.Println("Step 5: Testing the EvaluateMetrics function on the validation set...")
	exactAcc, closeAccs, proximity := bp.EvaluateMetrics(valData, valLabelsFloat)

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
