package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"phase" // Replace with your actual import path, e.g., "github.com/yourusername/phase"
)

type Sample struct {
	Inputs map[int]float64
	Label  int
}

const (
	baseURL        = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir       = "mnist_data"
	modelFile      = "saved_model.json"
	checkpointFile = "saved_checkpoint.json"
	modelDir       = "models"
)

var (
	initialCheckpoint []map[int]map[string]interface{}
	trainSize         int
	testSize          int
	trainSamples      []Sample
	testSamples       []Sample
	trainInputs       []map[int]float64
	trainLabels       []int
	processStartTime  time.Time
)

func main() {
	rand.Seed(time.Now().UnixNano())
	processStartTime = time.Now()
	fmt.Printf("Process started at %s\n", processStartTime.Format("2006-01-02 15:04:05"))
	setupMnist()
}

func setupMnist() {
	// Load MNIST dataset
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	fmt.Println("Step 2: Loading MNIST training dataset...")
	var err error
	trainInputs, trainLabels, err = loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(trainInputs))

	// Set trainSize to 80% of the total data
	trainSize = int(0.8 * float64(len(trainInputs)))

	// Calculate testSize as the remaining 20%
	testSize = len(trainInputs) - trainSize

	// Initialize trainSamples and testSamples slices
	trainSamples = make([]Sample, trainSize)
	testSamples = make([]Sample, testSize)

	// Populate trainSamples (first 80%)
	for i := 0; i < trainSize; i++ {
		trainSamples[i] = Sample{Inputs: trainInputs[i], Label: trainLabels[i]}
	}

	// Populate testSamples (remaining 20%)
	for i := 0; i < testSize; i++ {
		testSamples[i] = Sample{Inputs: trainInputs[trainSize+i], Label: trainLabels[trainSize+i]}
	}

	fmt.Printf("Using %d samples for training and %d samples for testing\n", len(trainSamples), len(testSamples))
}

func OLD() {
	// Seed random number generator.
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

	// Step 3: Always use the first five samples.
	fmt.Println("Step 3: Selecting the first five samples...")
	fiveInputs := trainInputs[:5]
	fiveLabels := trainLabels[:5]

	// Check if saved model and checkpoint exist.
	if fileExists(modelFile) && fileExists(checkpointFile) {
		fmt.Println("Saved model and checkpoint found. Loading them...")
		// Load the model.
		data, err := os.ReadFile(modelFile)
		if err != nil {
			log.Fatalf("Failed to read saved model: %v", err)
		}
		loadedBP := phase.NewPhase()
		if err := loadedBP.DeserializesFromJSON(string(data)); err != nil {
			log.Fatalf("Failed to deserialize model: %v", err)
		}
		// Load the checkpoint.
		chkData, err := os.ReadFile(checkpointFile)
		if err != nil {
			log.Fatalf("Failed to read saved checkpoint: %v", err)
		}
		var loadedCheckpoint []map[int]map[string]interface{}
		if err := json.Unmarshal(chkData, &loadedCheckpoint); err != nil {
			log.Fatalf("Failed to unmarshal checkpoint: %v", err)
		}
		// Run a cycle with the loaded model.
		runCycle("Cycle 4: Loaded Model", loadedBP, fiveInputs, fiveLabels, loadedCheckpoint)
	} else {
		// Otherwise, run the experiment normally.

		// Create the neural network.
		fmt.Println("Creating the neural network...")
		// For example: 784 inputs, 64 hidden neurons, and 10 output neurons.
		bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")

		// Compute the initial checkpoint (pre-output state) only once.
		fmt.Println("Computing initial checkpoint (pre-output states) for the first five samples...")
		initialCheckpoint := bp.CheckpointPreOutputNeurons(fiveInputs, 1)

		// Cycle 1: Initial state.
		runCycle("Cycle 1: Initial State", bp, fiveInputs, fiveLabels, initialCheckpoint)

		// Cycle 2: Add three neurons and then run both passes.
		fmt.Println("Adding first batch of three neurons...")
		for i := 0; i < 3; i++ {
			newNeuron := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
			if newNeuron == nil {
				log.Fatal("Failed to add new neuron!")
			}
			fmt.Printf("Added new neuron with ID %d\n", newNeuron.ID)
		}
		runCycle("Cycle 2: After First Batch", bp, fiveInputs, fiveLabels, initialCheckpoint)

		// Cycle 3: Add another three neurons and then run both passes.
		fmt.Println("Adding second batch of three neurons...")
		for i := 0; i < 3; i++ {
			newNeuron := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
			if newNeuron == nil {
				log.Fatal("Failed to add new neuron!")
			}
			fmt.Printf("Added new neuron with ID %d\n", newNeuron.ID)
		}
		runCycle("Cycle 3: After Second Batch", bp, fiveInputs, fiveLabels, initialCheckpoint)

		// --- Save model and checkpoint to disk.
		fmt.Println("Saving model and checkpoint to disk...")
		if err := bp.SaveToJSON(modelFile); err != nil {
			log.Fatalf("Failed to save model: %v", err)
		}
		chkData, err := json.MarshalIndent(initialCheckpoint, "", "  ")
		if err != nil {
			log.Fatalf("Failed to marshal checkpoint: %v", err)
		}
		if err := os.WriteFile(checkpointFile, chkData, 0644); err != nil {
			log.Fatalf("Failed to save checkpoint: %v", err)
		}
		fmt.Println("Model and checkpoint saved successfully.")

		// Optionally, run one more cycle with the saved state.
		runCycle("Cycle 4: Reloaded Model", bp, fiveInputs, fiveLabels, initialCheckpoint)
	}

	fmt.Println("Experiment complete.")
}

// runCycle runs a full forward pass (computing all neurons) and a partial forward pass
// that uses the saved checkpoint for the old neurons and computes only new neurons' outputs.
// It then prints the outputs and timing.
func runCycle(cycleLabel string, bp *phase.Phase, inputs []map[int]float64, labels []int, checkpoint []map[int]map[string]interface{}) {
	fmt.Println("\n==============================")
	fmt.Println(cycleLabel)
	fmt.Println("==============================")

	// Full Forward Pass.
	fmt.Println("\n--- Full Forward Pass Outputs ---")
	startFull := time.Now()
	for i, in := range inputs {
		bp.ResetNeuronValues()
		bp.Forward(in, 1)
		outputs := bp.GetOutputs()
		fmt.Printf("Sample %d (label %d): ", i+1, labels[i])
		for _, outID := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[outID])
		}
		fmt.Println()
	}
	fmt.Printf("Full Forward Pass Time: %v\n", time.Since(startFull))

	// Partial Forward Pass using the saved checkpoint.
	fmt.Println("\n--- Partial Forward Pass Outputs (using checkpoint) ---")
	startPartial := time.Now()
	for i, ck := range checkpoint {
		// ComputePartialOutputsFromCheckpoint should restore checkpointed state,
		// process only new neurons, and update the output neurons.
		outputs := bp.ComputePartialOutputsFromCheckpoint(ck)
		fmt.Printf("Sample %d (label %d): ", i+1, labels[i])
		for _, outID := range bp.OutputNodes {
			fmt.Printf("%.4f ", outputs[outID])
		}
		fmt.Println()
	}
	fmt.Printf("Partial Forward Pass Time: %v\n", time.Since(startPartial))
	time.Sleep(2 * time.Second)
}

// fileExists checks whether a file exists.
func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	return err == nil && !info.IsDir()
}

// --- Helper Functions for Data Loading and Downloads ---

func ensureMNISTDownloads(bp *phase.Phase, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []string{"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"}
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
	buf := make([]byte, 784) // 28x28
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
