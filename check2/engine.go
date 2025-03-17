package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"paragon" // Replace with your actual Paragon package import path
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

// Global variables for test data
var (
	testInputs [][][]float64
	testLabels []int
)

// ModelConfig holds configuration for each model variant
type ModelConfig struct {
	Name               string
	UseInMemory        bool
	UseFileCheckpoint  bool
	CheckpointLayerIdx int
}

// LoadModel loads a model if one exists; otherwise, it trains on MNIST
func LoadModel(config *ModelConfig, trainInputs, trainTargets, valInputs, valTargets, testInputs, testTargets [][][]float64) (*paragon.Network, error) {
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input layer
		{16, 16}, // Hidden layer
		{10, 1},  // Output layer
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	modelPath := filepath.Join(modelDir, modelFile)
	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("✅ Found pre-trained model for %s. Loading...\n", config.Name)
		if err := nn.LoadFromJSON(modelPath); err != nil {
			return nil, fmt.Errorf("failed to load model from JSON: %v", err)
		}
	} else {
		fmt.Printf("No pre-trained model found for %s; training new network...\n", config.Name)
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
		trainer.TrainWithValidation(trainInputs, trainTargets, valInputs, valTargets, testInputs, testTargets)
		if err := nn.SaveToJSON(modelPath); err != nil {
			return nil, fmt.Errorf("failed to save model to JSON: %v", err)
		}
		fmt.Println("✅ Model trained and saved.")
	}
	return nn, nil
}

// InferFull runs a full forward pass on the given sample
func InferFull(nn *paragon.Network, input [][]float64) ([]float64, time.Duration) {
	start := time.Now()
	nn.Forward(input)
	duration := time.Since(start)
	return extractOutput(nn), duration
}

// InferFromCheckpoint runs only the checkpoint portion
func InferFromCheckpoint(nn *paragon.Network, cp [][]float64, cpLayer int) ([]float64, time.Duration) {
	start := time.Now()
	nn.ForwardFromLayer(cpLayer, cp)
	duration := time.Since(start)
	return extractOutput(nn), duration
}

// InferFromFileCheckpoint handles file-based checkpoint inference
func InferFromFileCheckpoint(nn *paragon.Network, cp [][]float64, cpLayer int, configName string) ([]float64, time.Duration) {
	os.MkdirAll("checkpoints", os.ModePerm)
	tempFile := fmt.Sprintf("checkpoints/%s_%d_checkpoint_layer_%d.json", configName, time.Now().UnixNano(), cpLayer)
	data, err := json.Marshal(cp)
	if err != nil {
		log.Printf("Failed to marshal checkpoint for %s: %v", configName, err)
		return nil, 0
	}
	if err := os.WriteFile(tempFile, data, 0644); err != nil {
		log.Printf("Failed to write checkpoint file for %s: %v", configName, err)
		return nil, 0
	}
	loadedCp, err := nn.LoadLayerState(cpLayer, tempFile)
	if err != nil {
		log.Printf("Failed to load checkpoint for %s: %v", configName, err)
		return nil, 0
	}
	start := time.Now()
	nn.ForwardFromLayer(cpLayer, loadedCp)
	duration := time.Since(start)
	os.Remove(tempFile)
	return extractOutput(nn), duration
}

// extractOutput collects the output values from the network's output layer
func extractOutput(nn *paragon.Network) []float64 {
	outWidth := nn.Layers[nn.OutputLayer].Width
	output := make([]float64, outWidth)
	for x := 0; x < outWidth; x++ {
		output[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
	}
	return output
}

// computeAccuracy calculates accuracy given predicted and true labels
func computeAccuracy(predicted, trueLabels []int) float64 {
	if len(predicted) != len(trueLabels) {
		return 0
	}
	correct := 0
	for i := range predicted {
		if predicted[i] == trueLabels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(predicted))
}

// evaluateMethod prints accuracy, ADHD metrics, total time, and average time
func evaluateMethod(methodName string, acc float64, score float64, totalTime float64, avgTime float64) {
	fmt.Printf("%s - Accuracy: %.2f%%, ADHD Score: %.4f, Total Time: %.3f ms, Avg Time: %.3f ms\n",
		methodName, acc*100, score, totalTime, avgTime)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Ensure MNIST data is downloaded
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("Failed to download MNIST data: %v", err)
	}
	fmt.Println("MNIST Example with ADHD Comparison and Timing")

	// Load MNIST data
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}
	testLabels = make([]int, len(testTargets))
	for i, target := range testTargets {
		testLabels[i] = paragon.ArgMax(target[0])
	}
	trIn, trTarg, valIn, valTarg := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Printf("Train samples: %d, Validation samples: %d, Test samples: %d\n", len(trIn), len(valIn), len(testInputs))

	// Define model configurations
	fullConfig := ModelConfig{Name: "FullForward", UseInMemory: false, UseFileCheckpoint: false, CheckpointLayerIdx: 1}
	memConfig := ModelConfig{Name: "InMemoryCheckpoint", UseInMemory: true, UseFileCheckpoint: false, CheckpointLayerIdx: 1}
	fileConfig := ModelConfig{Name: "FileCheckpoint", UseInMemory: false, UseFileCheckpoint: true, CheckpointLayerIdx: 1}

	// Load models
	fullModel, err := LoadModel(&fullConfig, trIn, trTarg, valIn, valTarg, testInputs, testTargets)
	if err != nil {
		log.Fatalf("Failed to load full model: %v", err)
	}
	memModel := paragon.NewNetwork([]struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}},
		[]string{"leaky_relu", "leaky_relu", "softmax"}, []bool{true, false, true})
	if err := memModel.LoadFromJSON(filepath.Join(modelDir, modelFile)); err != nil {
		log.Fatalf("Failed to load in-memory model: %v", err)
	}
	fileModel := paragon.NewNetwork([]struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}},
		[]string{"leaky_relu", "leaky_relu", "softmax"}, []bool{true, false, true})
	if err := fileModel.LoadFromJSON(filepath.Join(modelDir, modelFile)); err != nil {
		log.Fatalf("Failed to load file checkpoint model: %v", err)
	}

	// Use a subset of test samples for evaluation
	numTestSamples := 100
	if numTestSamples > len(testInputs) {
		numTestSamples = len(testInputs)
	}
	testInputsSubset := testInputs[:numTestSamples]
	testLabelsSubset := testLabels[:numTestSamples]

	// Collect predictions, checkpoints, and timings
	predictedFull := make([]int, numTestSamples)
	predictedMem := make([]int, numTestSamples)
	predictedFile := make([]int, numTestSamples)
	checkpoints := make([][][]float64, numTestSamples)
	var fullTimes, memTimes, fileTimes []time.Duration

	for i, input := range testInputsSubset {
		// Full forward pass
		startFull := time.Now()
		fullModel.Forward(input)
		durationFull := time.Since(startFull)
		cpState := fullModel.GetLayerState(fullConfig.CheckpointLayerIdx)
		checkpoints[i] = cpState
		outputFull := extractOutput(fullModel)
		predictedFull[i] = paragon.ArgMax(outputFull)
		fullTimes = append(fullTimes, durationFull)

		// In-memory checkpoint
		startMem := time.Now()
		memModel.ForwardFromLayer(memConfig.CheckpointLayerIdx, cpState)
		durationMem := time.Since(startMem)
		outputMem := extractOutput(memModel)
		predictedMem[i] = paragon.ArgMax(outputMem)
		memTimes = append(memTimes, durationMem)

		// File-based checkpoint
		tempFile := fmt.Sprintf("checkpoints/temp_cp_%d.json", i)
		if err := fullModel.SaveLayerState(fullConfig.CheckpointLayerIdx, tempFile); err != nil {
			log.Printf("Failed to save checkpoint for sample %d: %v", i, err)
			continue
		}
		loadedCp, err := fileModel.LoadLayerState(fileConfig.CheckpointLayerIdx, tempFile)
		if err != nil {
			log.Printf("Failed to load checkpoint for sample %d: %v", i, err)
			continue
		}
		startFile := time.Now()
		fileModel.ForwardFromLayer(fileConfig.CheckpointLayerIdx, loadedCp)
		durationFile := time.Since(startFile)
		outputFile := extractOutput(fileModel)
		predictedFile[i] = paragon.ArgMax(outputFile)
		fileTimes = append(fileTimes, durationFile)
		os.Remove(tempFile)
	}

	// Compute average timings
	avgFullTime := averageDuration(fullTimes)
	avgMemTime := averageDuration(memTimes)
	avgFileTime := averageDuration(fileTimes)

	// Compute total timings
	totalFullTime := 0.0
	for _, d := range fullTimes {
		totalFullTime += float64(d.Nanoseconds()) / 1e6
	}
	totalMemTime := 0.0
	for _, d := range memTimes {
		totalMemTime += float64(d.Nanoseconds()) / 1e6
	}
	totalFileTime := 0.0
	for _, d := range fileTimes {
		totalFileTime += float64(d.Nanoseconds()) / 1e6
	}

	// Prepare expected outputs
	expectedOutputs := make([]float64, numTestSamples)
	for i, label := range testLabelsSubset {
		expectedOutputs[i] = float64(label)
	}

	// ADHD Evaluation for Full Forward
	actualOutputsFull := make([]float64, numTestSamples)
	for i, pred := range predictedFull {
		actualOutputsFull[i] = float64(pred)
	}
	fullModel.EvaluateModel(expectedOutputs, actualOutputsFull)
	accFull := computeAccuracy(predictedFull, testLabelsSubset)
	evaluateMethod("Full Forward", accFull, fullModel.Performance.Score, totalFullTime, avgFullTime)

	// ADHD Evaluation for In-Memory Checkpoint
	memModel.EvaluateFromCheckpoint(checkpoints, expectedOutputs, memConfig.CheckpointLayerIdx)
	accMem := computeAccuracy(predictedMem, testLabelsSubset)
	evaluateMethod("In-Memory Checkpoint", accMem, memModel.Performance.Score, totalMemTime, avgMemTime)

	// ADHD Evaluation for File Checkpoint
	fileModel.EvaluateFromCheckpoint(checkpoints, expectedOutputs, fileConfig.CheckpointLayerIdx)
	accFile := computeAccuracy(predictedFile, testLabelsSubset)
	evaluateMethod("File Checkpoint", accFile, fileModel.Performance.Score, totalFileTime, avgFileTime)

	// Verify outputs for Sample 0
	sample0 := testInputs[0]
	expectedLabel0 := testLabels[0]
	fmt.Printf("\n=== Testing Sample 0 (Label: %d) ===\n", expectedLabel0)

	fullOut0, _ := InferFull(fullModel, sample0)
	cp0 := fullModel.GetLayerState(fullConfig.CheckpointLayerIdx)
	memOut0, _ := InferFromCheckpoint(memModel, cp0, memConfig.CheckpointLayerIdx)
	fileOut0, _ := InferFromFileCheckpoint(fileModel, cp0, fileConfig.CheckpointLayerIdx, fileConfig.Name)

	fullPred0 := paragon.ArgMax(fullOut0)
	memPred0 := paragon.ArgMax(memOut0)
	filePred0 := paragon.ArgMax(fileOut0)

	fmt.Printf("FullForward - Prediction: %d, Output: %s\n", fullPred0, formatOutputHighPrecision(fullOut0))
	fmt.Printf("InMemoryCheckpoint - Prediction: %d, Output: %s\n", memPred0, formatOutputHighPrecision(memOut0))
	fmt.Printf("FileCheckpoint - Prediction: %d, Output: %s\n", filePred0, formatOutputHighPrecision(fileOut0))

	tolerance := 1e-6
	if slicesEqual(fullOut0, memOut0, tolerance) && slicesEqual(fullOut0, fileOut0, tolerance) {
		fmt.Println("✅ Sample 0 outputs match within tolerance.")
	} else {
		fmt.Println("❌ Sample 0 outputs do not match!")
		fmt.Printf("Max diff (full vs mem): %.17f\n", maxDifference(fullOut0, memOut0))
		fmt.Printf("Max diff (full vs file): %.17f\n", maxDifference(fullOut0, fileOut0))
	}

	fmt.Println("\nAll done!")
}

// Helper Functions (unchanged)
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

func downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	out, err := os.Create(filepath)
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

func loadMNISTData(dir string, isTraining bool) (inputs [][][]float64, targets [][][]float64, err error) {
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
	if rows != 28 || cols != 28 {
		return nil, nil, fmt.Errorf("unexpected image dimensions: %dx%d", rows, cols)
	}
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
	if magic := binary.BigEndian.Uint32(lblHeader[0:4]); magic != 2049 {
		return nil, nil, fmt.Errorf("invalid label magic number: %d", magic)
	}
	numLabels := int(binary.BigEndian.Uint32(lblHeader[4:8]))
	if numImages != numLabels {
		return nil, nil, fmt.Errorf("mismatch between number of images (%d) and labels (%d)", numImages, numLabels)
	}
	inputs = make([][][]float64, numImages)
	targets = make([][][]float64, numImages)
	imgBuf := make([]byte, 784)
	for i := 0; i < numImages; i++ {
		if _, err := fImg.Read(imgBuf); err != nil {
			return nil, nil, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		img := make([][]float64, 28)
		for r := 0; r < 28; r++ {
			img[r] = make([]float64, 28)
			for c := 0; c < 28; c++ {
				img[r][c] = float64(imgBuf[r*28+c]) / 255.0
			}
		}
		inputs[i] = img
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("failed to read label %d: %w", i, err)
		}
		label := int(lblByte[0])
		targets[i] = labelToTarget(label)
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

func slicesEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
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

func maxDifference(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	maxDiff := 0.0
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func averageDuration(durations []time.Duration) float64 {
	if len(durations) == 0 {
		return 0
	}
	total := 0.0
	for _, d := range durations {
		total += float64(d.Nanoseconds()) / 1e6 // Convert to milliseconds
	}
	return total / float64(len(durations))
}
