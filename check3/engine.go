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
	checkpointTrainDir   = "checkpoints/train"
	checkpointValDir     = "checkpoints/val"
)

func main() {
	// === Data Preparation ===
	// (Assume ensureMNISTDownloads, loadMNISTData, and SplitDataset are defined in your code.)
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
	trainSetInputs, trainSetTargets, valInputs, valTargets := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Printf("Training: %d, Validation: %d, Test: %d\n", len(trainSetInputs), len(valInputs), len(testInputs))

	// === Model Creation or Loading ===
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

	// Create copies for different evaluation methods.
	nnFull := nn
	nnCheckpoint := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	if err := nnCheckpoint.LoadFromJSON(modelPath); err != nil {
		log.Fatalf("Failed to load model into checkpoint copy: %v", err)
	}

	// === Checkpoint Generation ===
	// Generate a few sample checkpoints.
	if err := os.MkdirAll(checkpointSampleDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create sample checkpoint directory: %v", err)
	}
	fmt.Println("Handling sample checkpoint files...")
	for i := 0; i < numSampleCheckpoints; i++ {
		filename := filepath.Join(checkpointSampleDir, fmt.Sprintf("sample_cp_%d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			input := testInputs[i]
			nnFull.Forward(input)
			cpState := nnFull.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal sample checkpoint %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write sample checkpoint %d: %v", i, err)
			}
			fmt.Printf("Created sample checkpoint %d\n", i)
		}
	}

	// Create directories for full training and validation checkpoints.
	if err := os.MkdirAll(checkpointTrainDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create training checkpoint directory: %v", err)
	}
	if err := os.MkdirAll(checkpointValDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create validation checkpoint directory: %v", err)
	}
	fmt.Println("Generating checkpoint files for training and validation sets...")
	// For training set:
	for i := 0; i < len(trainSetInputs); i++ {
		filename := filepath.Join(checkpointTrainDir, fmt.Sprintf("sample_cp_%05d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			input := trainSetInputs[i]
			nnFull.Forward(input)
			cpState := nnFull.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal training checkpoint %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write training checkpoint %d: %v", i, err)
			}
		}
	}
	// For validation set:
	for i := 0; i < len(valInputs); i++ {
		filename := filepath.Join(checkpointValDir, fmt.Sprintf("sample_cp_%05d.json", i))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			input := valInputs[i]
			nnFull.Forward(input)
			cpState := nnFull.GetLayerState(checkpointLayerIdx)
			cpData, err := json.MarshalIndent(cpState, "", "  ")
			if err != nil {
				log.Fatalf("Failed to marshal validation checkpoint %d: %v", i, err)
			}
			if err := os.WriteFile(filename, cpData, 0644); err != nil {
				log.Fatalf("Failed to write validation checkpoint %d: %v", i, err)
			}
		}
	}

	// === Evaluation ===
	// For a single sample:
	sampleIndex := 0
	fmt.Printf("\n=== Full Forward Pass for Sample %d ===\n", sampleIndex)
	start := time.Now()
	nnFull.Forward(testInputs[sampleIndex])
	fullOut := extractOutput(nnFull)
	fullTime := time.Since(start)
	fmt.Printf("Full Forward Time: %v\n", fullTime)
	fmt.Printf("Full Output: %v\n", fullOut)

	fmt.Printf("\n=== Checkpoint-Based Forward Pass for Sample %d ===\n", sampleIndex)
	cpFile := filepath.Join(checkpointSampleDir, fmt.Sprintf("sample_cp_%d.json", sampleIndex))
	data, err := os.ReadFile(cpFile)
	if err != nil {
		log.Fatalf("Failed to read sample checkpoint: %v", err)
	}
	var cpState [][]float64
	if err := json.Unmarshal(data, &cpState); err != nil {
		log.Fatalf("Failed to unmarshal sample checkpoint: %v", err)
	}
	start = time.Now()
	nnCheckpoint.ForwardFromLayer(checkpointLayerIdx, cpState)
	cpOut := extractOutput(nnCheckpoint)
	cpTime := time.Since(start)
	fmt.Printf("Checkpoint Forward Time: %v\n", cpTime)
	fmt.Printf("Checkpoint Output: %v\n", cpOut)

	// Compare outputs.
	fmt.Println("\n=== Output Comparison ===")
	for i := 0; i < len(fullOut); i++ {
		diff := math.Abs(fullOut[i] - cpOut[i])
		if diff > 1e-6 {
			fmt.Printf("Neuron %d: Full=%.6f, Checkpoint=%.6f, Diff=%.6f\n", i, fullOut[i], cpOut[i], diff)
		}
	}

	// Evaluate on the training set using full forward pass.
	fmt.Println("\n=== ADHD Evaluation on Training Data (Full Forward) ===")
	start = time.Now()
	trainScoreFull := computeADHDScoreFull(nnFull, trainSetInputs, trainSetTargets)
	fmt.Printf("Training Set ADHD Score (Full Forward): %.4f, Time: %v\n", trainScoreFull, time.Since(start))

	// Evaluate on the training set using checkpoint-based evaluation.
	fmt.Println("\n=== ADHD Evaluation on Training Data (Checkpoint-Based) ===")
	trainCpFiles := sortedFilesInDir(checkpointTrainDir)
	// Create expected outputs from your ground truth targets.
	expectedTrain := make([]float64, len(trainSetTargets))
	for i, t := range trainSetTargets {
		expectedTrain[i] = float64(paragon.ArgMax(t[0]))
	}
	score, loadTime, forwardTime := nnCheckpoint.EvaluateFromCheckpointFilesWithTiming(trainCpFiles, expectedTrain, checkpointLayerIdx)
	fmt.Printf("Training Set ADHD Score (Checkpoint): %.4f, Total Load Time: %v, Total Forward Time: %v\n", score, loadTime, forwardTime)

	// Similarly, evaluate on the validation set.
	fmt.Println("\n=== ADHD Evaluation on Validation Data (Full Forward) ===")
	start = time.Now()
	valScoreFull := computeADHDScoreFull(nnFull, valInputs, valTargets)
	fmt.Printf("Validation Set ADHD Score (Full Forward): %.4f, Time: %v\n", valScoreFull, time.Since(start))

	fmt.Println("\n=== ADHD Evaluation on Validation Data (Checkpoint-Based) ===")
	valCpFiles := sortedFilesInDir(checkpointValDir)
	expectedVal := make([]float64, len(valTargets))
	for i, t := range valTargets {
		expectedVal[i] = float64(paragon.ArgMax(t[0]))
	}
	score, loadTime, forwardTime = nnCheckpoint.EvaluateFromCheckpointFilesWithTiming(valCpFiles, expectedVal, checkpointLayerIdx)
	fmt.Printf("Validation Set ADHD Score (Checkpoint): %.4f, Total Load Time: %v, Total Forward Time: %v\n", score, loadTime, forwardTime)
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

// computeADHDScoreFull computes the ADHD score using a full forward pass
func computeADHDScoreFull(nn *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	var expected, actual []float64
	for i, input := range inputs {
		nn.Forward(input)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(targets[i][0])
		expected = append(expected, float64(trueLabel))
		actual = append(actual, float64(pred))
	}
	nn.EvaluateModel(expected, actual)
	return nn.Performance.Score
}

// computeADHDScoreCheckpoint computes the ADHD score using checkpoint files
func computeADHDScoreCheckpoint(nn *paragon.Network, checkpointFiles []string, targets [][][]float64, layerIdx int) float64 {
	var expected, actual []float64
	for i, cpFile := range checkpointFiles {
		data, err := os.ReadFile(cpFile)
		if err != nil {
			log.Printf("Failed to read checkpoint file %s: %v", cpFile, err)
			continue
		}
		var cpState [][]float64
		if err := json.Unmarshal(data, &cpState); err != nil {
			log.Printf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
			continue
		}
		nn.ForwardFromLayer(layerIdx, cpState)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(targets[i][0])
		expected = append(expected, float64(trueLabel))
		actual = append(actual, float64(pred))
	}
	nn.EvaluateModel(expected, actual)
	return nn.Performance.Score
}

// computeADHDScoreCheckpointInMemory evaluates ADHD score by first
// capturing an in-memory checkpoint state at checkpointLayerIdx for each sample,
// then running the remainder of the forward pass.
func computeADHDScoreCheckpointInMemory(nn *paragon.Network, inputs [][][]float64, targets [][][]float64, checkpointLayerIdx int) float64 {
	var expected, actual []float64
	for i, input := range inputs {
		// First, run a full forward pass on a copy (or use your full network)
		nn.Forward(input)
		// Get checkpoint state from the desired layer
		cpState := nn.GetLayerState(checkpointLayerIdx)
		// Now run the forward pass from that checkpoint state using a network copy.
		// (It is important to use a separate network or reset the network to avoid interference.)
		nn.ForwardFromLayer(checkpointLayerIdx, cpState)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(targets[i][0])
		expected = append(expected, float64(trueLabel))
		actual = append(actual, float64(pred))
	}
	nn.EvaluateModel(expected, actual)
	return nn.Performance.Score
}

// computeADHDScoreFromFileCheckpoints evaluates the ADHD score by loading checkpoint files
// (which were generated beforehand) and running the forward pass from the checkpoint layer.
// It also accumulates the file load time and the forward pass time.
func computeADHDScoreFromFileCheckpoints(nn *paragon.Network, checkpointFiles []string, targets [][][]float64, checkpointLayerIdx int) float64 {
	var expected, actual []float64
	var totalLoadTime, totalForwardTime time.Duration

	for i, cpFile := range checkpointFiles {
		startLoad := time.Now()
		data, err := os.ReadFile(cpFile)
		if err != nil {
			log.Printf("Failed to read checkpoint file %s: %v", cpFile, err)
			continue
		}
		var cpState [][]float64
		if err := json.Unmarshal(data, &cpState); err != nil {
			log.Printf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
			continue
		}
		totalLoadTime += time.Since(startLoad)

		startForward := time.Now()
		nn.ForwardFromLayer(checkpointLayerIdx, cpState)
		totalForwardTime += time.Since(startForward)

		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(targets[i][0])
		expected = append(expected, float64(trueLabel))
		actual = append(actual, float64(pred))
	}
	nn.EvaluateModel(expected, actual)
	fmt.Printf("Checkpoint file load time: %v, forward time: %v\n", totalLoadTime, totalForwardTime)
	return nn.Performance.Score
}
