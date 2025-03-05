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

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// **Step 1: Download MNIST if needed**
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// **Step 2: Load MNIST training data**
	fmt.Println("Step 2: Loading MNIST training dataset...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(trainInputs))

	// **Step 3: Create the neural network**
	fmt.Println("Step 3: Creating the neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")

	// **Step 4: Select five random samples**
	fmt.Println("Step 4: Selecting five random samples...")
	indices := make([]int, 5)
	for i := 0; i < 5; i++ {
		indices[i] = rand.Intn(len(trainInputs))
	}
	var fiveInputs []map[int]float64
	var fiveLabels []int
	for _, idx := range indices {
		fiveInputs = append(fiveInputs, trainInputs[idx])
		fiveLabels = append(fiveLabels, trainLabels[idx])
	}

	// **Step 5: Create checkpoints for the five samples**
	fmt.Println("Step 5: Creating checkpoints for the five samples...")
	fiveCheckpoints := bp.CheckpointPreOutputNeurons(fiveInputs, 1)

	// **Step 6: Compute and print outputs before adding neurons**
	fmt.Println("Step 6: Outputs before adding neurons:")
	for i, ck := range fiveCheckpoints {
		outputs := bp.ComputeOutputsFromCheckpoint(ck)
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
	}

	// **Step 7: Add three new neurons just before the output neurons**
	fmt.Println("\nStep 7: Adding three new neurons...")
	for i := 0; i < 3; i++ {
		newN := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
		if newN == nil {
			log.Fatal("Failed to add new neuron!")
		}
		fmt.Printf("Added new neuron %d\n", newN.ID)
	}

	// **Step 8: Rerun checkpoints and print outputs after adding neurons**
	fmt.Println("Step 8: Outputs after adding neurons:")
	for i, ck := range fiveCheckpoints {
		outputs := bp.ComputeOutputsFromCheckpoint(ck)
		fmt.Printf("Sample %d (label %d): ", i+1, fiveLabels[i])
		for _, id := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[id])
		}
		fmt.Println()
	}

	fmt.Println("\nDone. Observe the slight variational changes in the outputs due to the added neurons.")
}

// **Helper Functions**

// ensureMNISTDownloads downloads and unzips MNIST files if they don't exist
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

// loadMNIST loads MNIST images and labels from the specified files
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

	// Prepare slices
	inputs := make([]map[int]float64, limit)
	labels := make([]int, limit)

	// Buffer for one image (784 = 28×28)
	buf := make([]byte, 784)
	for i := 0; i < limit; i++ {
		// Read image
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data at sample %d: %w", i, err)
		}
		// Build input map
		inputMap := make(map[int]float64, 784)
		for px := 0; px < 784; px++ {
			inputMap[px] = float64(buf[px]) / 255.0 // Normalize [0..255] -> [0..1]
		}
		inputs[i] = inputMap

		// Read label
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data at sample %d: %w", i, err)
		}
		labels[i] = int(lblByte[0])
	}
	return inputs, labels, nil
}
