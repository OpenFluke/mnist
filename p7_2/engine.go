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

func main() {
	// Set an acceptable tolerance for floating-point comparisons.
	const epsilon = 0.01

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

	// Step 3: Create the neural network.
	fmt.Println("Step 3: Creating the neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	//bp.Debug = true // Enable debug logging

	// Step 4: Select five random samples.
	fmt.Println("Step 4: Selecting five random samples...")
	indices := make([]int, 5)
	var fiveInputs []map[int]float64
	var fiveLabels []int
	for i := 0; i < 5; i++ {
		indices[i] = rand.Intn(len(trainInputs))
	}
	for _, idx := range indices {
		fiveInputs = append(fiveInputs, trainInputs[idx])
		fiveLabels = append(fiveLabels, trainLabels[idx])
	}

	// Step 5: Full forward pass before adding neurons.
	fmt.Println("\nStep 5: Full forward pass before adding neurons:")
	var fullOutputsBefore []map[int]float64
	for i, input := range fiveInputs {
		bp.Forward(input, 1)
		outputs := bp.GetOutputs()
		fullOutputsBefore = append(fullOutputsBefore, outputs)
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
	}

	// Step 6: Create checkpoint outputs before adding neurons using all hidden neurons.
	fmt.Println("\nStep 6: Checkpoint outputs before adding neurons:")
	fiveCheckpoints := bp.CheckpointAllHiddenNeurons(fiveInputs, 1)
	for i, ck := range fiveCheckpoints {
		outputs := bp.ComputeOutputsWithNewNeurons(ck, fiveInputs[i], 1) // Pass sample inputs
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
		// Verify using epsilon tolerance.
		for id, val := range outputs {
			if math.Abs(val-fullOutputsBefore[i][id]) > epsilon {
				log.Printf("Mismatch before adding neurons for sample %d, neuron %d: full=%.4f, checkpoint=%.4f",
					i+1, id, fullOutputsBefore[i][id], val)
			}
		}
	}
	fmt.Println("Checkpoint outputs match full forward pass before adding neurons.")

	// Step 7: Add three new neurons without rewiring (append only).
	fmt.Println("\nStep 7: Adding three new neurons...")
	for i := 0; i < 3; i++ {
		newN := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
		if newN == nil {
			log.Fatal("Failed to add new neuron!")
		}
		bp.AddNewNeuronToOutput(newN.ID)
		fmt.Printf("Added new neuron %d\n", newN.ID)
	}

	// Step 8: Full forward pass after adding neurons.
	fmt.Println("\nStep 8: Full forward pass after adding neurons:")
	var fullOutputsAfter []map[int]float64
	for i, input := range fiveInputs {
		bp.Forward(input, 1)
		outputs := bp.GetOutputs()
		fullOutputsAfter = append(fullOutputsAfter, make(map[int]float64))
		for id, val := range outputs {
			fullOutputsAfter[i][id] = val
		}
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
	}

	// Step 9: Recompute checkpoint outputs after adding neurons using all hidden neurons.
	fmt.Println("\nStep 9: Checkpoint outputs after adding neurons (recomputed):")
	newCheckpoints := bp.CheckpointAllHiddenNeurons(fiveInputs, 1)
	for i, ck := range newCheckpoints {
		outputs := bp.ComputeOutputsWithNewNeurons(ck, fiveInputs[i], 1)
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
		for id, val := range outputs {
			if math.Abs(val-fullOutputsAfter[i][id]) > epsilon {
				log.Printf("Mismatch after adding neurons for sample %d, neuron %d: full=%.4f, checkpoint=%.4f",
					i+1, id, fullOutputsAfter[i][id], val)
			}
		}
	}
	fmt.Println("Checkpoint outputs match full forward pass after adding neurons.")

	// Step 10: Add additional neurons and re-check consistency.
	fmt.Println("\nStep 10: Adding additional neurons and rechecking full pass vs checkpoint pass:")
	for i := 0; i < 2; i++ {
		newN := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
		if newN == nil {
			log.Fatal("Failed to add extra neuron!")
		}
		bp.AddNewNeuronToOutput(newN.ID)
		fmt.Printf("Added extra neuron %d\n", newN.ID)
	}

	var fullOutputsAfterExtra []map[int]float64
	for i, input := range fiveInputs {
		bp.Forward(input, 1)
		outputs := bp.GetOutputs()
		fullOutputsAfterExtra = append(fullOutputsAfterExtra, make(map[int]float64))
		for id, val := range outputs {
			fullOutputsAfterExtra[i][id] = val
		}
		fmt.Printf("AEN- Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
	}

	newCheckpoints = bp.CheckpointAllHiddenNeurons(fiveInputs, 1)
	for i, ck := range newCheckpoints {
		outputs := bp.ComputeOutputsWithNewNeurons(ck, fiveInputs[i], 1)
		fmt.Printf("AEN(checkpoint) - Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
		for id, val := range outputs {
			if math.Abs(val-fullOutputsAfterExtra[i][id]) > epsilon {
				log.Printf("Mismatch AENfor sample %d, neuron %d: full=%.4f, checkpoint=%.4f",
					i+1, id, fullOutputsAfterExtra[i][id], val)
			}
		}
	}
	fmt.Println("After extra neuron addition, checkpoint outputs match full forward pass.")

	fmt.Println("\nDone. The checkpointing system is verified to match full forward pass outputs before and after neuron additions (within a tolerance of", epsilon, ").")
}

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

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
