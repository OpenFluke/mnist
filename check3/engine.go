package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"paragon"
)

const (
	baseURL              = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir             = "mnist_data"
	modelDir             = "models"
	modelFile            = "mnist_model.json"
	checkpointSampleDir  = "checkpoints/sample" // Directory for sample checkpoints
	checkpointLayerIdx   = 1                    // Layer index for checkpointing
	numSampleCheckpoints = 5                    // Number of sample checkpoints
)

func main() {
	// ### Data Preparation
	// Download and prepare the MNIST dataset if not already present
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST downloads: %v", err)
	}
	fmt.Println("MNIST data ready.")

	// Load training and test data
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	// Split training data (80% for training, though we only need the full set here)
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Printf("Training samples: %d, Test samples: %d\n", len(trainSetInputs), len(testInputs))

	// ### Model Creation or Loading
	// Define network architecture
	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	modelPath := filepath.Join(modelDir, modelFile)
	var nn *paragon.Network
	if _, err := os.Stat(modelPath); err == nil {
		// Load existing model
		fmt.Println("Pre-trained model found. Loading model...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
		if err := nn.LoadFromJSON(modelPath); err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
	} else {
		// Train and save a new model
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

	// Create two copies of the model
	nnFull := nn // For full forward pass
	nnCheckpoint := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	if err := nnCheckpoint.LoadFromJSON(modelPath); err != nil {
		log.Fatalf("Failed to load model into nnCheckpoint: %v", err)
	}

	// ### Create or Load 5 Sample Checkpoints
	// Create directory for sample checkpoints
	if err := os.MkdirAll(checkpointSampleDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create sample checkpoint directory: %v", err)
	}

	// Handle the first 5 test samples for checkpointing
	fmt.Println("Handling 5 sample checkpoint files...")
	for i := 0; i < numSampleCheckpoints; i++ {
		filename := filepath.Join(checkpointSampleDir, fmt.Sprintf("sample_cp_%d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			// Create checkpoint if it doesn't exist
			input := testInputs[i]
			nnFull.Forward(input)
			cpState := nnFull.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal sample checkpoint for sample %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write sample checkpoint file for sample %d: %v", i, err)
			}
			fmt.Printf("Created checkpoint for sample %d\n", i)
		} else {
			fmt.Printf("Checkpoint for sample %d already exists, skipping creation\n", i)
		}
	}

	// ### Comparison of Full Pass vs Checkpoint for a Single Sample
	// Select sample index 0 for comparison
	// Select sample index for comparison
	sampleIndex := 0
	input := testInputs[sampleIndex]
	cpFile := filepath.Join(checkpointSampleDir, fmt.Sprintf("sample_cp_%d.json", sampleIndex))

	// Ensure checkpoint exists
	if _, err := os.Stat(cpFile); os.IsNotExist(err) {
		log.Fatalf("Checkpoint file does not exist for sample %d", sampleIndex)
	}

	// **Full Forward Pass**
	fmt.Printf("\n=== Full Forward Pass for Sample %d ===\n", sampleIndex)
	startFull := time.Now()
	nnFull.Forward(input)
	endFull := time.Now()
	fullTime := endFull.Sub(startFull)

	// Extract and display outputs
	fullOut := extractOutput(nnFull)
	fmt.Printf("Full Forward Time: %v\n", fullTime)
	fmt.Println("Neuron Outputs (Full Forward):")
	for i, val := range fullOut {
		fmt.Printf("Neuron %d: %.6f\n", i, val)
	}

	// **Checkpoint-Based Forward Pass**
	fmt.Printf("\n=== Checkpoint-Based Forward Pass for Sample %d ===\n", sampleIndex)

	// Load checkpoint with timing
	startLoad := time.Now()
	data, err := os.ReadFile(cpFile)
	if err != nil {
		log.Fatalf("Failed to read checkpoint file %s: %v", cpFile, err)
	}
	var cpState [][]float64
	if err := json.Unmarshal(data, &cpState); err != nil {
		log.Fatalf("Failed to unmarshal checkpoint: %v", err)
	}
	endLoad := time.Now()
	loadTime := endLoad.Sub(startLoad)

	// Perform forward pass from checkpoint with timing
	startCP := time.Now()
	nnCheckpoint.ForwardFromLayer(checkpointLayerIdx, cpState)
	endCP := time.Now()
	cpTime := endCP.Sub(startCP)

	// Extract and display outputs
	cpOut := extractOutput(nnCheckpoint)
	fmt.Printf("Checkpoint Load Time: %v\n", loadTime)
	fmt.Printf("Checkpoint Forward Time: %v\n", cpTime)
	fmt.Println("Neuron Outputs (Checkpoint):")
	for i, val := range cpOut {
		fmt.Printf("Neuron %d: %.6f\n", i, val)
	}

	// **Compare Outputs**
	fmt.Println("\n=== Output Comparison ===")
	differencesFound := false
	for i := 0; i < len(fullOut); i++ {
		diff := math.Abs(fullOut[i] - cpOut[i])
		if diff > 1e-6 {
			fmt.Printf("Neuron %d: Full=%.6f, Checkpoint=%.6f, Difference=%.6f\n",
				i, fullOut[i], cpOut[i], diff)
			differencesFound = true
		}
	}
	if !differencesFound {
		fmt.Println("No significant differences found between outputs.")
	}
	fmt.Println("Evaluation complete.")
}

// ## Helper Functions

// ### ensureMNISTDownloads
// Ensures the MNIST dataset is downloaded and extracted
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

// ### downloadFile
// Downloads a file from a URL to the specified path
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

// ### unzipFile
// Extracts a gzipped file to the destination path
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

// ### loadMNISTData
// Loads MNIST images and labels from the specified directory
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
	_ = int(binary.BigEndian.Uint32(lblHeader[4:8]))
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

// ### labelToTarget
// Converts a label to a one-hot encoded target
func labelToTarget(label int) [][]float64 {
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	if label >= 0 && label < 10 {
		target[0][label] = 1.0
	}
	return target
}

// ### extractOutput
// Extracts the output layer values from the network
func extractOutput(nn *paragon.Network) []float64 {
	outWidth := nn.Layers[nn.OutputLayer].Width
	output := make([]float64, outWidth)
	for x := 0; x < outWidth; x++ {
		output[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
	}
	return output
}

// ### computeAccuracy
// Calculates accuracy from predicted and true labels
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

// ### sortedFilesInDir
// Returns a sorted list of file paths from a directory
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

func formatOutputHighPrecision(output []float64) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, val := range output {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("%.17f", val))
	}
	sb.WriteString("]")
	return sb.String()
}
